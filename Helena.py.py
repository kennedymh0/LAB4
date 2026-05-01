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
observe_queue = queue.Queue()
save_queue = queue.Queue()
stop_event = threading.Event()

# ===============================================================
# USER SETTINGS
# ===============================================================

# ---- Map region (High Velocity Cloud project) ----
L_MIN = 90
L_MAX = 180
B_MIN = 20
B_MAX = 60
STEP = 2

# ---- Telescope ----
LAT = 37.8732

# ---- SDR settings ----
CENTER_FREQ = 1420.405e6      # HI line
SAMPLE_RATE = 2.4e6
GAIN = 40

NSAMPLES = 2048                # samples per FFT
NBLOCKS = 500                  # blocks per capture
N_AVG_SKY = 175                # averages per pointing
N_AVG_CAL = 12

DATA_DIR = "HVC_data"          # Directory to save individual observation files

# ===============================================================
# GLOBAL MATRICES (Calculated once to save CPU)
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
    
    # 1. Finish the current row (b1) starting from the user's l1
    l1_step = 2.0 / np.cos(np.radians(b1))
    for q in np.arange(l1, L_MAX + l1_step, l1_step):
        targets.append((q, b1))

    # 2. Map all SUBSEQUENT rows, starting from b1 + STEP
    for b in np.arange(b1 + STEP, B_MAX + STEP, STEP):  
        l_step = 2.0 / np.cos(np.radians(b))
        
        for l in np.arange(L_MIN, L_MAX + l_step, l_step):
            targets.append((l, b))

    return targets

# ===============================================================
# GALACTIC -> ALTAZ
# ===============================================================

def gal_to_altaz(l_deg, b_deg):

    l = np.radians(l_deg)
    b = np.radians(b_deg)

    x = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])

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

    if az < 0:
        az += 360

    return alt, az

# ===============================================================
# SDR SETUP
# ===============================================================

def setup_sdr(index):
    sdr = ugradio.sdr.SDR(direct=False, device_index=index)
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN
    return sdr

# ===============================================================
# DATA CAPTURE AND SPECTRUM
# ===============================================================

def capture_and_average(nobs, sdr):
    """Capture data and compute averaged power spectrum on-the-fly to save RAM."""
    try:
        # Pre-compute window and initialize running sum
        window = np.hanning(NSAMPLES)
        running_sum = np.zeros(NSAMPLES)
        
        for i in range(nobs):
            x = sdr.capture_data(NSAMPLES, NBLOCKS)
            
            # Convert raw IQ data to complex numbers
            x_complex = x[..., 0] + 1j * x[..., 1]
            
            # Remove DC offset per block
            x_complex -= np.mean(x_complex, axis=1, keepdims=True)
            
            # Apply Hann Window
            x_complex = x_complex * window
            
            # Calculate the FFT over the samples (axis 1) and compute power
            spec = np.abs(np.fft.fftshift(np.fft.fft(x_complex, axis=1), axes=1)) ** 2
            
            # Add the averaged blocks for this capture to our running total
            running_sum += np.mean(spec, axis=0)
            
        # Return the final averaged spectrum directly
        return running_sum / nobs

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 ERROR IN SDR CAPTURE THREAD 🚨")
        print(f"Hardware or memory error: {e}")
        traceback.print_exc()
        print("="*50 + "\n")
        return None

# ===============================================================
# THREAD 1 : POINTING
# ===============================================================

def pointing_thread(scope, targets):

    for l_deg, b_deg in targets:
        if stop_event.is_set():
            break

        try:
            alt, az = gal_to_altaz(l_deg, b_deg)

            if alt < 15 or alt > 85:
                print("Skipping l={}, b={} (alt={:.1f})".format(l_deg, b_deg, alt))
                continue

            print("\nPointing to l={}, b={}".format(l_deg, b_deg))
            print("alt={:.1f}, az={:.1f}".format(alt, az))

            # scope.point blocks until slew is complete (wait=True is default)
            scope.point(alt, az)

            observe_queue.put({
                "l": l_deg,
                "b": b_deg,
                "alt": alt,
                "az": az
            })

            observe_queue.join()

        except Exception as e:
            print("Pointing error:", e)

    stop_event.set()

# ===============================================================
# THREAD 2 : OBSERVER
# ===============================================================

def observer_thread(sdr0, sdr1, noise):
    
    # Calculate frequencies once
    freqs = np.fft.fftshift(
        np.fft.fftfreq(NSAMPLES, d=1.0/SAMPLE_RATE)
    ) + CENTER_FREQ

    # Use a ThreadPoolExecutor to handle the SDRs concurrently without re-spawning threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while not stop_event.is_set():

            try:
                target = observe_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                l_deg = target["l"]
                b_deg = target["b"]

                # --- Sky observation (noise diode off) ---
                noise.off()
                
                # Submit jobs to the thread pool
                future_sky0 = executor.submit(capture_and_average, N_AVG_SKY, sdr0)
                future_sky1 = executor.submit(capture_and_average, N_AVG_SKY, sdr1)
                
                # .result() blocks until the thread finishes and returns the data
                power0 = future_sky0.result()
                power1 = future_sky1.result()

                # --- Calibration observation (noise diode on) ---
                noise.on()
                
                future_cal0 = executor.submit(capture_and_average, N_AVG_CAL, sdr0)
                future_cal1 = executor.submit(capture_and_average, N_AVG_CAL, sdr1)
                
                pcal0 = future_cal0.result()
                pcal1 = future_cal1.result()

                noise.off()

                if None in (power0, power1, pcal0, pcal1):
                    print("Skipping save for l={}, b={} due to SDR failure".format(l_deg, b_deg))
                    observe_queue.task_done()
                    continue

                save_queue.put({
                    "time": time.time(),
                    "l": l_deg,
                    "b": b_deg,
                    "alt": target["alt"],
                    "az": target["az"],
                    "freq": freqs,
                    "power0": power0,
                    "power1": power1,
                    "pcal0": pcal0,
                    "pcal1": pcal1,
                })

                print("Observed l={}, b={}".format(l_deg, b_deg))

                observe_queue.task_done()

            except Exception as e:
                print("Observer error:", e)

# ===============================================================
# THREAD 3 : WRITER
# ===============================================================

def writer_thread():
    
    # Ensure our data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    saved_count = 0

    while not stop_event.is_set() or not save_queue.empty():

        try:
            item = save_queue.get(timeout=1)
            
            # Format filename nicely, e.g., HVC_data/HVC_l120.0_b30.0.npz
            filename = os.path.join(DATA_DIR, f"HVC_l{item['l']:.1f}_b{item['b']:.1f}.npz")
            
            # Save this specific observation as its own file
            np.savez(filename, **item)
            
            saved_count += 1
            print("Saved observation {} to {}".format(saved_count, filename))

        except queue.Empty:
            continue

        except Exception as e:
            print("Writer error:", e)

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
    print("{} targets loaded.".format(len(targets)))

    threads = [
        threading.Thread(target=pointing_thread,
                         args=(scope, targets),
                         daemon=True),

        threading.Thread(target=observer_thread,
                         args=(sdr0, sdr1, noise),
                         daemon=True),

        threading.Thread(target=writer_thread,
                         daemon=True)
    ]

    for t in threads:
        t.start()

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
        except:
            pass
            
        try:
            print("Closing SDRs...")
            sdr0.close()
            sdr1.close()
        except:
            pass

        print("Done.")

# ===============================================================
# RUN
# ===============================================================

if __name__ == "__main__":
    main()