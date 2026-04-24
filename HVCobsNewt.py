import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time
import ugradio
import traceback

# ===============================================================
# GLOBAL THREADING OBJECTS
# ===============================================================
sdr0queue = queue.Queue()
sdr1queue = queue.Queue()
observe_queue = queue.Queue()
save_queue = queue.Queue()
stop_event = threading.Event()

# ===============================================================
# USER SETTINGS
# ===============================================================

# ---- Map region (High Velocity Cloud project) ----
L_MIN = 60
L_MAX = 180
B_MIN = 20
B_MAX = 60
STEP = 2

# ---- Telescope ----
LAT = 37.8732

# ---- SDR settings ----
CENTER_FREQ = 1420.405751e6      # HI line
SAMPLE_RATE = 2.4e6
GAIN = 40

NSAMPLES = 2048                # samples per FFT
NBLOCKS = 50                    # blocks per capture (was undefined)
N_AVG_SKY = 1750                      # averages per pointing
N_AVG_CAL = 350
SETTLE_TIME = 20                 # sec after move

OUTFILE = "HVC_map.npz"

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
    for b in np.arange(b1 + STEP, B_MAX + STEP, STEP):  # <--- Start at b1 + STEP
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

    R_eq_to_gal = np.array([
        [-0.054876, -0.873437, -0.483835],
        [ 0.494109, -0.444830,  0.746982],
        [-0.867666, -0.198076,  0.455984]
    ])

    R_gal_to_eq = np.linalg.inv(R_eq_to_gal)

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


def capture_and_average(nobs, sdr, result_queue):
    """Capture data and compute averaged power spectrum on-the-fly to save RAM."""
    try:
        specs = []
        
        # Pre-compute the Hann window once to save CPU time
        window = np.hanning(NSAMPLES)
        
        for i in range(nobs):
            x = sdr.capture_data(NSAMPLES, NBLOCKS)
            
            # Convert raw IQ data to complex numbers
            x_complex = x[..., 0] + 1j * x[..., 1]
            
            # 1. Remove DC offset (center the signal around 0)
            x_complex -= np.mean(x_complex)
            
            # 2. APPLY HANN WINDOW: Taper the edges of the time-domain blocks
            x_complex = x_complex * window
            
            # 3. Calculate the FFT and compute power
            spec = np.abs(np.fft.fftshift(np.fft.fft(x_complex))) ** 2
            
            # Average the blocks in this capture together and save
            specs.append(np.mean(spec, axis=0))
            
            time.sleep(0.001)
            
        # Average all the captures and put ONE final spectrum into the queue
        result_queue.put(np.mean(specs, axis=0))

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 ERROR IN SDR CAPTURE THREAD 🚨")
        print(f"Hardware or memory error: {e}")
        traceback.print_exc() # Prints the exact line number where it crashed
        print("="*50 + "\n")
        
        # Put None in the queue so the observer thread doesn't hang forever
        result_queue.put(None)

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

            print("Pointing to l={}, b={}".format(l_deg, b_deg))
            print("alt={:.1f}, az={:.1f}".format(alt, az))

            scope.point(alt, az)

            time.sleep(SETTLE_TIME)

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

    while not stop_event.is_set():

        # FIX: use timeout so stop_event can unblock this thread
        try:
            target = observe_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            l_deg = target["l"]   # FIX: read coords from the target dict,
            b_deg = target["b"]   # not undefined local variables

           # --- Sky observation (noise diode off) ---
            noise.off()

            q0_sky, q1_sky = queue.Queue(), queue.Queue()
            
            # Use N_AVG_SKY for the 2.5 minute integration
            thread0 = threading.Thread(target=capture_and_average, args=(N_AVG_SKY, sdr0, q0_sky))
            thread1 = threading.Thread(target=capture_and_average, args=(N_AVG_SKY, sdr1, q1_sky))
            thread0.start()
            thread1.start()
            thread0.join()
            thread1.join()
            
            # Retrieve the already-calculated spectrum
            power0 = q0_sky.get(timeout=180)   
            power1 = q1_sky.get(timeout=180)

            # --- Calibration observation (noise diode on) ---
            noise.on()

            q0_cal, q1_cal = queue.Queue(), queue.Queue()
            
            # Use N_AVG_CAL for the 30-second integration
            thread0 = threading.Thread(target=capture_and_average, args=(N_AVG_CAL, sdr0, q0_cal))
            thread1 = threading.Thread(target=capture_and_average, args=(N_AVG_CAL, sdr1, q1_cal))
            thread0.start()
            thread1.start()
            thread0.join()
            thread1.join()
            
            pcal0 = q0_cal.get(timeout=60)
            pcal1 = q1_cal.get(timeout=60)

            noise.off()

            #gain0 = (pcal0 - power0) / 79  # calibration can be separate
            #gain1 = (pcal1 - power1) / 58

            #Ta0 = power0 / gain0
            #Ta1 = power1 / gain1

            #final_spec = 0.5 * (Ta0 + Ta1)

            # FIX: freqs derived from SDR centre frequency and sample rate
            freqs = np.fft.fftshift(
                np.fft.fftfreq(NSAMPLES, d=1.0/SAMPLE_RATE)
            ) + CENTER_FREQ

            # FIX: indentation corrected; try/except now wraps the whole block
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

    all_data = []

    while not stop_event.is_set() or not save_queue.empty():

        try:
            item = save_queue.get(timeout=1)

            all_data.append(item)

            np.savez(OUTFILE, data=np.array(all_data, dtype=object))

            print("Saved {} observations".format(len(all_data)))

        except queue.Empty:
            continue

        except Exception as e:
            print("Writer error:", e)

# ===============================================================
# MAIN
# ===============================================================

def main():
    print("Please input starting galactic coordinate (l, degrees): ")
    l_start = float(input())  # <--- Add float() here!
    print("Please input starting galactic coordinate (b, degrees): ")
    b_start = float(input())  # <--- Add float() here!
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
