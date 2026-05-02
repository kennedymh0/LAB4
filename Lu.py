import os
import numpy as np
import time
import threading
import queue
import ugradio
import traceback
import multiprocessing as mp

# ===============================================================
# GLOBAL THREADING & MULTIPROCESSING OBJECTS
# ===============================================================
target_queue = queue.Queue()           # Passes targets to the observer
math_save_queue = queue.Queue()        # Passes reduced data to the background writer thread
capture_done_event = threading.Event() # Signals the telescope to start moving again
stop_event = threading.Event()

# ===============================================================
# USER SETTINGS
# ===============================================================

L_MIN, L_MAX = 90, 180
B_MIN, B_MAX = 20, 60
STEP = 2
LAT = 37.8732

CENTER_FREQ = 1420.405e6      
SAMPLE_RATE = 2.4e6
GAIN = 40

NSAMPLES = 2048                
NBLOCKS = 500                  
N_AVG_SKY = 150                 
N_AVG_CAL = 12

DATA_DIR = "HVC_new_data"          

# ===============================================================
# GLOBAL MATRICES
# ===============================================================
R_eq_to_gal = np.array([
    [-0.054876, -0.873437, -0.483835],
    [ 0.494109, -0.444830,  0.746982],
    [-0.867666, -0.198076,  0.455984]
])
R_gal_to_eq = np.linalg.inv(R_eq_to_gal)

# ===============================================================
# BUILD TARGET LIST
# ===============================================================

def build_targets(l1, b1):
    targets = []
    l1_step = 2.0 / np.cos(np.radians(b1))
    for q in np.arange(l1, L_MAX + l1_step, l1_step):
        targets.append((q, b1))

    for b in np.arange(b1 + STEP, B_MAX + STEP, STEP):  
        l_step = 2.0 / np.cos(np.radians(b))
        for l in np.arange(L_MIN, L_MAX + l_step, l_step):
            targets.append((l, b))

    return targets

def gal_to_altaz(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    x = np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
    lst = ugradio.timing.lst()
    phi = np.radians(LAT)

    R_eq_to_ha = np.array([
        [ np.cos(lst),  np.sin(lst), 0],
        [ np.sin(lst), -np.cos(lst), 0],
        [ 0,            0,           1]
    ])

    R_ha_to_azalt = np.array([
        [-np.sin(phi), 0, np.cos(phi)],
        [0, -1, 0],
        [np.cos(phi), 0, np.sin(phi)]
    ])

    R = np.matmul(R_ha_to_azalt, np.matmul(R_eq_to_ha, R_gal_to_eq))
    xp = np.matmul(R, x)

    az = np.degrees(np.arctan2(xp[1], xp[0]))
    alt = np.degrees(np.arcsin(xp[2]))

    if az < 0: az += 360
    return alt, az

def setup_sdr(index):
    sdr = ugradio.sdr.SDR(direct=False, device_index=index)
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN
    return sdr

# ===============================================================
# MULTIPROCESSING: SDR WORKER
# ===============================================================

def sdr_worker(device_index, cmd_queue, res_queue, math_res_queue):
    """
    Handles data capture and offloads the FFT to an internal background thread 
    so massive numpy arrays never cross the multiprocessing boundary.
    """
    import ugradio 
    import numpy as np
    import threading
    import queue
    
    try:
        sdr = setup_sdr(device_index)
        print(f"[WORKER {device_index}] SDR initialized.")
    except Exception as e:
        print(f"[WORKER {device_index}] Failed to init SDR: {e}")
        return

    # Internal queue just for this worker's math thread
    math_q = queue.Queue()

    def internal_math_thread():
        window = np.hanning(NSAMPLES)
        while True:
            task = math_q.get()
            if task is None: break
            massive_capture, nobs, task_type = task
            
            try:
                # Use reshape to avoid copying data with np.split
                reshaped = massive_capture.reshape(nobs, NBLOCKS, NSAMPLES, 2)
                running_sum = np.zeros(NSAMPLES)
                
                for i in range(nobs):
                    raw = reshaped[i]
                    x_complex = raw[..., 0] + 1j * raw[..., 1]
                    x_complex -= np.mean(x_complex, axis=1, keepdims=True)
                    x_complex *= window
                    spec = np.abs(np.fft.fftshift(np.fft.fft(x_complex, axis=1), axes=1)) ** 2
                    running_sum += np.mean(spec, axis=0)
                
                result = running_sum / nobs
                # Send the tiny reduced spectrum back to main process
                math_res_queue.put({"status": "ok", "type": task_type, "data": result})
            except Exception as e:
                math_res_queue.put({"status": "error", "error": str(e)})
            finally:
                # Strictly enforce garbage collection of the 1.2 GB object
                del massive_capture
				if 'reshaped' in locals():
					del reshaped
				math_q.task_done()

    # Start the internal math thread
    threading.Thread(target=internal_math_thread, daemon=True).start()

    while True:
        cmd = cmd_queue.get()
        if cmd["action"] == "stop":
            math_q.put(None)
            sdr.close()
            print(f"[WORKER {device_index}] SDR closed.")
            break
        elif cmd["action"] == "capture":
            nobs = cmd["nobs"]
            task_type = cmd["type"]
            try:
                total_blocks = nobs * NBLOCKS
                massive_capture = sdr.capture_data(NSAMPLES, total_blocks)
                
                # TELL OBSERVER IMMEDIATELY THAT HARDWARE CAPTURE IS DONE
                res_queue.put({"status": "captured"})
                
                # Pass data to the local math thread so SDR is ready for next command
                math_q.put((massive_capture, nobs, task_type))
            except Exception as e:
                res_queue.put({"status": "error", "error": str(e)})

# ===============================================================
# THREAD 1 : POINTING
# ===============================================================

def pointing_thread(scope, targets):
    for l_deg, b_deg in targets:
        if stop_event.is_set(): break

        try:
            alt, az = gal_to_altaz(l_deg, b_deg)
            if alt < 15 or alt > 85 or az < 5 or az > 350:
                print(f"Skipping l={l_deg}, b={b_deg} (alt={alt:.1f})")
                continue

            print(f"\n[TELESCOPE] Slewing to l={l_deg}, b={b_deg} (alt={alt:.1f}, az={az:.1f})")
            scope.point(alt, az)

            capture_done_event.clear()
            target_queue.put({"l": l_deg, "b": b_deg, "alt": alt, "az": az})

            capture_done_event.wait()

        except Exception as e:
            print("[TELESCOPE] Error:", e)

    stop_event.set()

# ===============================================================
# THREAD 2 : COORDINATOR OBSERVER
# ===============================================================

def observer_thread(cmd_q0, cmd_q1, res_q0, res_q1, math_res_q0, math_res_q1, noise):
    freqs = np.fft.fftshift(np.fft.fftfreq(NSAMPLES, d=1.0/SAMPLE_RATE)) + CENTER_FREQ

    while not stop_event.is_set():
        try:
            target = target_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            l_deg, b_deg = target["l"], target["b"]
            print(f"[OBSERVER] Capturing parallel raw data for l={l_deg}, b={b_deg}...")
            
            # --- Sky Capture ---
            noise.off()
            cmd_q0.put({"action": "capture", "nobs": N_AVG_SKY, "type": "sky"})
            cmd_q1.put({"action": "capture", "nobs": N_AVG_SKY, "type": "sky"})
            
            # Wait for physical capture to finish
            if res_q0.get()["status"] == "error" or res_q1.get()["status"] == "error":
                raise Exception("Sky Capture failed in worker.")

            # --- Cal Capture ---
            noise.on()
            cmd_q0.put({"action": "capture", "nobs": N_AVG_CAL, "type": "cal"})
            cmd_q1.put({"action": "capture", "nobs": N_AVG_CAL, "type": "cal"})

            if res_q0.get()["status"] == "error" or res_q1.get()["status"] == "error":
                raise Exception("Cal Capture failed in worker.")
            noise.off()

            # SIGNAL TELESCOPE TO MOVE IMMEDIATELY
            capture_done_event.set()
            target_queue.task_done()

            # --- Wait for Math (Happens in background while slewing!) ---
            results0, results1 = {}, {}
            for _ in range(2): # Expecting 2 math returns per worker (sky and cal)
                msg0 = math_res_q0.get()
                msg1 = math_res_q1.get()
                
                if msg0["status"] == "error" or msg1["status"] == "error":
                    raise Exception("Math thread crashed inside worker.")
                    
                results0[msg0["type"]] = msg0["data"]
                results1[msg1["type"]] = msg1["data"]

            # Send purely the reduced spectra to the writer thread (virtually zero RAM cost)
            math_save_queue.put({
                "time": time.time(),
                "l": l_deg, "b": b_deg, "alt": target["alt"], "az": target["az"],
                "freq": freqs,
                "power0": results0["sky"], "power1": results1["sky"],
                "pcal0": results0["cal"], "pcal1": results1["cal"]
            })

        except Exception as e:
            print("[OBSERVER] Error:", e)
            capture_done_event.set() # Prevent deadlock if capture fails

# ===============================================================
# THREAD 3 : WRITER (BACKGROUND)
# ===============================================================

def writer_thread():
    """Now purely a file I/O thread. No math happens here anymore."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    saved_count = 0
    
    while not stop_event.is_set() or not math_save_queue.empty():
        try:
            item = math_save_queue.get(timeout=1)
            
            filename = os.path.join(DATA_DIR, f"HVC_l{item['l']:.1f}_b{item['b']:.1f}.npz")
            np.savez(filename,
                     time=item['time'], l=item['l'], b=item['b'],
                     alt=item['alt'], az=item['az'], freq=item['freq'],
                     power0=item['power0'], power1=item['power1'], 
                     pcal0=item['pcal0'], pcal1=item['pcal1'])
            
            saved_count += 1
            print(f"[WRITER] Saved observation {saved_count}: {filename}")
            math_save_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print("[WRITER] Error:", e)

# ===============================================================
# MAIN
# ===============================================================

def main():
    print("Please input starting galactic coordinate (l, degrees): ")
    l_start = float(input())
    print("Please input starting galactic coordinate (b, degrees): ")
    b_start = float(input())
    
    print("Initializing telescope...")
    scope = ugradio.leusch.LeuschTelescope()
    noise = ugradio.leusch.LeuschNoise()
    noise.off()

    print("Initializing SDR workers...")
    cmd_q0, res_q0, math_res_q0 = mp.Queue(), mp.Queue(), mp.Queue()
    cmd_q1, res_q1, math_res_q1 = mp.Queue(), mp.Queue(), mp.Queue()

    p0 = mp.Process(target=sdr_worker, args=(0, cmd_q0, res_q0, math_res_q0), daemon=True)
    p1 = mp.Process(target=sdr_worker, args=(1, cmd_q1, res_q1, math_res_q1), daemon=True)
    p0.start()
    p1.start()

    targets = build_targets(l1=l_start, b1=b_start)
    print(f"{len(targets)} targets loaded.")

    threads = [
        threading.Thread(target=pointing_thread, args=(scope, targets), daemon=True),
        threading.Thread(target=observer_thread, args=(cmd_q0, cmd_q1, res_q0, res_q1, math_res_q0, math_res_q1, noise), daemon=True),
        threading.Thread(target=writer_thread, daemon=True)
    ]

    print("FINISHED ALL THREADS; SAVING NOW")

    for t in threads: t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping observation...")
        stop_event.set()
    finally:
        try:
            print("Stowing telescope...")
            scope.stow()
        except: pass
        
        print("Sending stop commands to SDR workers...")
        cmd_q0.put({"action": "stop"})
        cmd_q1.put({"action": "stop"})
        
        p0.join(timeout=3)
        p1.join(timeout=3)
        
        print("Done.")

if __name__ == "__main__":
    main()
