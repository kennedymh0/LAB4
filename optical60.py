import os
import sys
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
# We will initialize these in reset_globals() so they are fresh on restart.
target_queue = None           
math_save_queue = None        
metadata_queue = None
capture_done_event = None 
stop_event = None
restart_requested = None

# Watchdog tracking
activity_timestamp = 0
activity_lock = threading.Lock()
current_l = 0.0
current_b = 0.0

def reset_globals():
    """Generates fresh queues and events for a clean restart."""
    global target_queue, math_save_queue, metadata_queue, capture_done_event, stop_event, restart_requested
    target_queue = queue.Queue()
    math_save_queue = queue.Queue()
    metadata_queue = queue.Queue()
    capture_done_event = threading.Event()
    stop_event = threading.Event()
    restart_requested = threading.Event()

def update_activity():
    """Updates the global timestamp to prevent the watchdog from barking."""
    global activity_timestamp
    with activity_lock:
        activity_timestamp = time.time()

# ===============================================================
# USER SETTINGS
# ===============================================================

L_MIN, L_MAX = 60, 90
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

    math_q = queue.Queue()

    def internal_math_thread():
        window = np.hanning(NSAMPLES)
        while True:
            task = math_q.get()
            if task is None: break
            massive_capture, nobs, task_type = task
            
            try:
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
                math_res_queue.put({"status": "ok", "type": task_type, "data": result})
            except Exception as e:
                math_res_queue.put({"status": "error", "error": str(e)})
            finally:
                del massive_capture
                if 'reshaped' in locals():
                    del reshaped
                math_q.task_done()

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
                res_queue.put({"status": "captured"})
                math_q.put((massive_capture, nobs, task_type))
                del massive_capture 

            except Exception as e:
                res_queue.put({"status": "error", "error": str(e)})

# ===============================================================
# THREADS
# ===============================================================

def watchdog_thread():
    """Monitors activity. If stalled for > 120 seconds, triggers a hard restart."""
    global activity_timestamp, current_l, current_b
    while not stop_event.is_set():
        time.sleep(5)
        with activity_lock:
            elapsed = time.time() - activity_timestamp
            
        if elapsed > 120:
            print(f"\n[WATCHDOG] SYSTEM STALL DETECTED (No activity for {elapsed:.0f}s). TRIGGERING RESTART!")
            restart_requested.set()
            stop_event.set()  # Signals all other threads to die
            break

def pointing_thread(scope, targets):
    global current_l, current_b
    for l_deg, b_deg in targets:
        if stop_event.is_set(): break

        try:
            alt, az = gal_to_altaz(l_deg, b_deg)
            if alt < 15 or alt > 85 or az < 5 or az > 350:
                print(f"Skipping l={l_deg}, b={b_deg} (alt={alt:.1f})")
                continue
            
            # Record current target for watchdog restart
            current_l, current_b = l_deg, b_deg

            print(f"\n[TELESCOPE] Slewing to l={l_deg}, b={b_deg} (alt={alt:.1f}, az={az:.1f})")
            update_activity()
            scope.point(alt, az)
            update_activity()

            capture_done_event.clear()
            target_queue.put({"l": l_deg, "b": b_deg, "alt": alt, "az": az})

            capture_done_event.wait()

        except Exception as e:
            print("[TELESCOPE] Error:", e)

    # Only set stop event naturally if we actually finished all targets
    if not restart_requested.is_set():
        stop_event.set()

def observer_thread(cmd_q0, cmd_q1, res_q0, res_q1, noise):
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
            update_activity()
            cmd_q0.put({"action": "capture", "nobs": N_AVG_SKY, "type": "sky"})
            cmd_q1.put({"action": "capture", "nobs": N_AVG_SKY, "type": "sky"})
            
            if res_q0.get()["status"] == "error" or res_q1.get()["status"] == "error":
                raise Exception("Sky Capture failed in worker.")

            # --- Cal Capture ---
            noise.on()
            update_activity()
            cmd_q0.put({"action": "capture", "nobs": N_AVG_CAL, "type": "cal"})
            cmd_q1.put({"action": "capture", "nobs": N_AVG_CAL, "type": "cal"})

            if res_q0.get()["status"] == "error" or res_q1.get()["status"] == "error":
                raise Exception("Cal Capture failed in worker.")
            noise.off()
            update_activity()

            # SIGNAL TELESCOPE TO MOVE IMMEDIATELY
            capture_done_event.set()
            target_queue.task_done()

            # --- Pass off to Collector ---
            metadata_queue.put({
                "time": time.time(),
                "l": l_deg, "b": b_deg, "alt": target["alt"], "az": target["az"]
            })

        except Exception as e:
            print("[OBSERVER] Error:", e)
            capture_done_event.set()

def writer_thread():
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
            update_activity()
            math_save_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print("[WRITER] Error:", e)

def collector_thread(math_res_q0, math_res_q1):
    freqs = np.fft.fftshift(np.fft.fftfreq(NSAMPLES, d=1.0/SAMPLE_RATE)) + CENTER_FREQ
    
    while not stop_event.is_set():
        try:
            meta = metadata_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        try:
            results0, results1 = {}, {}
            for _ in range(2): 
                msg0 = math_res_q0.get()
                msg1 = math_res_q1.get()
                
                if msg0["status"] == "error" or msg1["status"] == "error":
                    print("[COLLECTOR] Math thread crashed inside worker.")
                    continue
                    
                results0[msg0["type"]] = msg0["data"]
                results1[msg1["type"]] = msg1["data"]

            math_save_queue.put({
                "time": meta["time"],
                "l": meta["l"], "b": meta["b"], "alt": meta["alt"], "az": meta["az"],
                "freq": freqs,
                "power0": results0["sky"], "power1": results1["sky"],
                "pcal0": results0["cal"], "pcal1": results1["cal"]
            })
            metadata_queue.task_done()
        except Exception as e:
            print("[COLLECTOR] Error:", e)

# ===============================================================
# CORE OBSERVATION WRAPPER
# ===============================================================

def run_observation_session(l_start, b_start):
    """Runs the main processes. Returns (needs_restart, next_l, next_b)."""
    global current_l, current_b
    
    # Initialize fresh queues and events for this session
    reset_globals()
    current_l, current_b = l_start, b_start
    update_activity()
    
    print(f"Starting session at L={l_start}, B={b_start}")
    
    print("Initializing telescope...")
    try:
        scope = ugradio.leusch.LeuschTelescope()
        noise = ugradio.leusch.LeuschNoise()
        noise.off()
    except Exception as e:
        print(f"Failed to connect to telescope/noise: {e}")
        return True, current_l, current_b # Instantly trigger restart logic if hardware connection fails

    print("Initializing SDR workers...")
    cmd_q0, res_q0, math_res_q0 = mp.Queue(), mp.Queue(), mp.Queue()
    cmd_q1, res_q1, math_res_q1 = mp.Queue(), mp.Queue(), mp.Queue()

    p0 = mp.Process(target=sdr_worker, args=(0, cmd_q0, res_q0, math_res_q0), daemon=True)
    p1 = mp.Process(target=sdr_worker, args=(1, cmd_q1, res_q1, math_res_q1), daemon=True)
    p0.start()
    p1.start()

    targets = build_targets(l1=current_l, b1=current_b)
    print(f"{len(targets)} targets loaded.")

    threads = [
        threading.Thread(target=watchdog_thread, daemon=True), # NEW WATCHDOG
        threading.Thread(target=pointing_thread, args=(scope, targets), daemon=True),
        threading.Thread(target=observer_thread, args=(cmd_q0, cmd_q1, res_q0, res_q1, noise), daemon=True),
        threading.Thread(target=collector_thread, args=(math_res_q0, math_res_q1), daemon=True),
        threading.Thread(target=writer_thread, daemon=True)
    ]

    print("FINISHED ALL THREADS; SAVING NOW")

    for t in threads: t.start()

    try:
        # Wait for either completion, keyboard stop, or watchdog triggering the stop_event
        while any(t.is_alive() for t in threads) and not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping observation (Manual override)...")
        stop_event.set()
    finally:
        # If we exited naturally (no restart requested), stow the telescope.
        if not restart_requested.is_set():
            try:
                print("Stowing telescope...")
                scope.stow()
            except: pass
        else:
            print("Restart requested: Skipping telescope stow.")
        
        print("Sending stop commands to SDR workers...")
        try:
            cmd_q0.put({"action": "stop"})
            cmd_q1.put({"action": "stop"})
        except Exception:
            pass # Queues might be closed
        
        # Give them 3 seconds to close gracefully
        p0.join(timeout=3)
        p1.join(timeout=3)
        
        # HARD KILL if they are locked up
        if p0.is_alive():
            print("[SYSTEM] Force terminating SDR 0 process...")
            p0.terminate()
        if p1.is_alive():
            print("[SYSTEM] Force terminating SDR 1 process...")
            p1.terminate()
        
        print("Observation session cleaned up.")
        
    return restart_requested.is_set(), current_l, current_b

# ===============================================================
# MAIN
# ===============================================================

def main():
    print("Please input starting galactic coordinate (l, degrees): ")
    l_start = float(input())
    print("Please input starting galactic coordinate (b, degrees): ")
    b_start = float(input())
    
    while True:
        needs_restart, l_start, b_start = run_observation_session(l_start, b_start)
        
        if not needs_restart:
            print("Observations fully complete. Exiting script.")
            break
            
        print("\n" + "="*60)
        print(f"   WATCHDOG RESTARTING ENTIRE SYSTEM at l={l_start:.1f}, b={b_start:.1f}   ")
        print("="*60 + "\n")
        time.sleep(3) # Give OS a moment to free hardware ports

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Recommended for robust SDR multiprocessing
    main()
