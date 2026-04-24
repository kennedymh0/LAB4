import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# USER SETTINGS
# ===============================================================
DATA_FILE = "hvctest.npz"

def main():
    print(f"Loading data from {DATA_FILE}...")
    try:
        # allow_pickle=True is required because the data is an array of dicts
        archive = np.load(DATA_FILE, allow_pickle=True)
        obs_data = archive['data']
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}. Check your file path.")
        return
    except KeyError:
        print("Error: 'data' key not found in the .npz file.")
        return

    if len(obs_data) == 0:
        print("The data file is empty. No observations to plot.")
        return

    print(f"Loaded {len(obs_data)} observations.")

    # ===============================================================
    # 1. PREPARE TIME SERIES DATA
    # ===============================================================
    # Extract times and convert to minutes elapsed since the first observation
    times = [obs["time"] for obs in obs_data]
    t0 = times[0]
    rel_times_minutes = [(t - t0) / 60.0 for t in times]

    # Calculate the mean sky power across all frequency bins for each pointing
    mean_power0 = [np.mean(obs["power0"]) for obs in obs_data]
    mean_power1 = [np.mean(obs["power1"]) for obs in obs_data]

    # ===============================================================
    # 2. PREPARE CALIBRATED SPECTRUM DATA
    # ===============================================================
    # Let's look at the very first observation for the spectrum plot
    obs = obs_data[0]
    
    # Convert frequencies from Hz to MHz for easier reading
    freqs_mhz = obs["freq"] / 1e6

    # Calculate gain using the noise diode calibration info
    # Pol 0 increments by 79 K, Pol 1 increments by 58 K
    gain0 = (obs["pcal0"] - obs["power0"]) / 79.0
    gain1 = (obs["pcal1"] - obs["power1"]) / 58.0

    # Avoid division by zero or negative gains in case of RFI/glitches
    gain0[gain0 <= 0] = np.nan
    gain1[gain1 <= 0] = np.nan

    # Calculate Antenna Temperature (Ta) in Kelvin
    Ta0 = obs["power0"] / gain0
    Ta1 = obs["power1"] / gain1

    # ===============================================================
    # 3. PLOT
    # ===============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Top Plot: Time Series ---
    ax1.plot(rel_times_minutes, mean_power0, label="Pol 0 (Sky)", marker='o', linestyle='-', markersize=4)
    ax1.plot(rel_times_minutes, mean_power1, label="Pol 1 (Sky)", marker='x', linestyle='-', markersize=4)
    ax1.set_xlabel("Time since start (minutes)")
    ax1.set_ylabel("Mean Raw Power")
    ax1.set_title("Time Series: System Stability (Mean Power per Pointing)")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # --- Bottom Plot: Calibrated Spectrum ---
    ax2.plot(freqs_mhz, Ta0, label="Pol 0 ($T_A$)", alpha=0.8)
    ax2.plot(freqs_mhz, Ta1, label="Pol 1 ($T_A$)", alpha=0.8)
    
    # Mark the theoretical 21-cm HI line center (1420.405751 MHz)
    ax2.axvline(1420.405751, color='red', linestyle='--', alpha=0.6, label="HI Rest Freq")
    
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Antenna Temperature (K)")
    ax2.set_title(f"Calibrated Spectrum for Pointing: l={obs['l']:.1f}°, b={obs['b']:.1f}°")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()