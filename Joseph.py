import os
import numpy as np
import time
import threading
import queue
import ugradio
import traceback
import concurrent.futures

# ===============================================================
# GLOBAL THREADING OBJECTS
# ===============================================================
target_queue = queue.Queue()       # Passes targets to the observer
math_save_queue = queue.Queue()    # Passes raw data to the background math thread
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
N_AVG_SKY = 150                 # REDUCED: Speeds up capture by 2x
N_AVG_CAL = 12

DATA_DIR = "HVC_data"          

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
# THREAD 1 : POINTING
# ===============================================================

def pointing_thread(scope, targets):
    for l_deg, b_deg in targets:
        if stop_event.is_set(): break

        try:
            alt, az = gal_to_altaz(l_deg, b_deg)
            if alt < 15 or alt > 85:
                print(f"Skipping l={l_deg}, b={b_deg} (alt={alt:.1f})")
                continue

            print(f"\n[TELESCOPE] Slewing to l={l_deg}, b={b_deg} (alt={alt:.1f}, az={az:.1f})")
            scope.point(alt, az)

            # Signal the observer thread that slewing is done
            capture_done_event.clear()
            target_queue.put({"l": l_deg, "b": b_deg, "alt": alt, "az": az})

            # Wait ONLY for raw data capture to finish, NOT the math
            capture_done_event.wait()

        except Exception as e:
            print("[TELESCOPE] Error:", e)

    stop_event.set()

# ===============================================================
# THREAD 2 : RAW DATA OBSERVER
# ===============================================================

def capture_raw(nobs, sdr):
    """Captures raw IQ data as fast as possible with no math."""
    raw_blocks = []
    for _ in range(nobs):
        raw_blocks.append(sdr.capture_data(NSAMPLES, NBLOCKS))
    return raw_blocks

def observer_thread(sdr0, sdr1, noise):
    # Frequencies computed once
    freqs = np.fft.fftshift(np.fft.fftfreq(NSAMPLES, d=1.0/SAMPLE_RATE)) + CENTER_FREQ

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while not stop_event.is_set():
            try:
                target = target_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                l_deg, b_deg = target["l"], target["b"]
                print(f"[OBSERVER] Capturing pure raw data for l={l_deg}, b={b_deg}...")
                
                # --- Sky Capture ---
                noise.off()
                future_sky0 = executor.submit(capture_raw, N_AVG_SKY, sdr0)
                future_sky1 = executor.submit(capture_raw, N_AVG_SKY, sdr1)
                raw_sky0, raw_sky1 = future_sky0.result(), future_sky1.result()

                # --- Cal Capture ---
                noise.on()
                future_cal0 = executor.submit(capture_raw, N_AVG_CAL, sdr0)
                future_cal1 = executor.submit(capture_raw, N_AVG_CAL, sdr1)
                raw_cal0, raw_cal1 = future_cal0.result(), future_cal1.result()
                noise.off()

                # SIGNAL TELESCOPE TO MOVE IMMEDIATELY
                capture_done_event.set()
                target_queue.task_done()

                # Send raw data to background math thread
                math_save_queue.put({
                    "time": time.time(),
                    "l": l_deg, "b": b_deg, "alt": target["alt"], "az": target["az"],
                    "freq": freqs,
                    "raw_sky0": raw_sky0, "raw_sky1": raw_sky1,
                    "raw_cal0": raw_cal0, "raw_cal1": raw_cal1
                })

            except Exception as e:
                print("[OBSERVER] Error:", e)
                capture_done_event.set() # Prevent deadlock if capture fails

# ===============================================================
# THREAD 3 : MATH AND WRITER (BACKGROUND)
# ===============================================================

def process_spectrum(raw_data_list):
    """Computes FFT iteratively to keep RAM usage extremely low."""
    window = np.hanning(NSAMPLES)
    running_sum = np.zeros(NSAMPLES)
    
    for raw in raw_data_list:
        # raw shape: (NBLOCKS, NSAMPLES, 2)
        x_complex = raw[..., 0] + 1j * raw[..., 1]
        x_complex -= np.mean(x_complex, axis=1, keepdims=True)
        x_complex *= window
        spec = np.abs(np.fft.fftshift(np.fft.fft(x_complex, axis=1), axes=1)) ** 2
        running_sum += np.mean(spec, axis=0)
        
    return running_sum / len(raw_data_list)

def math_writer_thread():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    saved_count = 0
    while not stop_event.is_set() or not math_save_queue.empty():
        try:
            item = math_save_queue.get(timeout=1)
            
            print(f"[MATH] Computing FFTs for l={item['l']}, b={item['b']} in background...")
            
            power0 = process_spectrum(item['raw_sky0'])
            power1 = process_spectrum(item['raw_sky1'])
            pcal0 = process_spectrum(item['raw_cal0'])
            pcal1 = process_spectrum(item['raw_cal1'])

            filename = os.path.join(DATA_DIR, f"HVC_l{item['l']:.1f}_b{item['b']:.1f}.npz")
            np.savez(filename,
                     time=item['time'], l=item['l'], b=item['b'],
                     alt=item['alt'], az=item['az'], freq=item['freq'],
                     power0=power0, power1=power1, pcal0=pcal0, pcal1=pcal1)
            
            saved_count += 1
            print(f"[WRITER] ✅ Saved observation {saved_count}: {filename}")
            math_save_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print("[MATH/WRITER] Error:", e)

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

    print("Initializing SDRs...")
    sdr0 = setup_sdr(0)
    sdr1 = setup_sdr(1)

    targets = build_targets(l1=l_start, b1=b_start)
    print(f"{len(targets)} targets loaded.")

    threads = [
        threading.Thread(target=pointing_thread, args=(scope, targets), daemon=True),
        threading.Thread(target=observer_thread, args=(sdr0, sdr1, noise), daemon=True),
        threading.Thread(target=math_writer_thread, daemon=True)
    ]

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
        try:
            print("Closing SDRs...")
            sdr0.close()
            sdr1.close()
        except: pass
        print("Done.")

if __name__ == "__main__":
    main()