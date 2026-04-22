import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time
import ugradio

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

NSAMPLES = 2**18                # samples per FFT
N_AVG = 20                      # averages per pointing

SETTLE_TIME = 5                 # sec after move

OUTFILE = "HVC_map.npz"

# ===============================================================
# BUILD TARGET LIST
# ===============================================================

def build_targets():
    targets = []

    for b in np.arange(B_MIN, B_MAX + STEP, STEP):
        for l in np.arange(L_MIN, L_MAX + STEP, STEP):
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

    R = R_ha_to_azalt @ R_eq_to_ha @ R_gal_to_eq

    xp = R @ x

    az = np.degrees(np.arctan2(xp[1], xp[0]))
    alt = np.degrees(np.arcsin(xp[2]))

    if az < 0:
        az += 360

    return alt, az

# ===============================================================
# SDR SETUP
# ===============================================================

def setup_sdr(index):

    sdr = ugradio.sdr.SDR(device_index=index)

    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN

    return sdr

# ===============================================================
# GET POWER SPECTRUM
# ===============================================================

def get_spectrum(sdr):

    x = sdr.capture_data(NSAMPLES)

    spec = np.abs(np.fft.fftshift(np.fft.fft(x)))**2

    return spec

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
                print(f"Skipping l={l_deg}, b={b_deg} (alt={alt:.1f})")
                continue

            print(f"Pointing to l={l_deg}, b={b_deg}")
            print(f"alt={alt:.1f}, az={az:.1f}")

            scope.point(alt, az)

            time.sleep(SETTLE_TIME)

            observe_queue.put({
                "l": l_deg,
                "b": b_deg,
                "alt": alt,
                "az": az
            })

        except Exception as e:
            print("Pointing error:", e)

    stop_event.set()

# ===============================================================
# THREAD 2 : OBSERVER
# ===============================================================

def observer_thread(sdr0, sdr1):
def observer_thread(sdr0, sdr1, noise):

    while not stop_event.is_set():

        target = observe_queue.get()

        noise.off()
        time.sleep(1)

        sky0 = get_spectrum(sdr0)
        sky1 = get_spectrum(sdr1)

        noise.on()
        time.sleep(1)

        cal0 = get_spectrum(sdr0)
        cal1 = get_spectrum(sdr1)

        noise.off()

        gain0 = (cal0 - sky0)/79
        gain1 = (cal1 - sky1)/58

        Ta0 = sky0/gain0
        Ta1 = sky1/gain1

        final_spec = 0.5*(Ta0 + Ta1)

            save_queue.put({
                "time": time.time(),
                "l": l_deg,
                "b": b_deg,
                "alt": target["alt"],
                "az": target["az"],
                "freq": freqs,
                "spectrum": final_spec
            })

            print(f"Observed l={l_deg}, b={b_deg}")

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

            print(f"Saved {len(all_data)} observations")

        except queue.Empty:
            continue

        except Exception as e:
            print("Writer error:", e)

# ===============================================================
# MAIN
# ===============================================================

def main():

    print("Initializing telescope...")

    scope = ugradio.leusch.LeuschTelescope()
    noise = ugradio.leusch.LeuschNoise()
    noise.off()

    print("Initializing SDRs...")

    sdr0 = setup_sdr(0)
    sdr1 = setup_sdr(1)

    targets = build_targets()

    print(f"{len(targets)} targets loaded.")

    scope_ =

    threads = [
        threading.Thread(target=pointing_thread,
                         args=(scope, targets),
                         daemon=True),

        threading.Thread(target=observer_thread,
                         args=(sdr0, sdr1),
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

        print("Done.")

# ===============================================================
# RUN
# ===============================================================

if __name__ == "__main__":
    main()