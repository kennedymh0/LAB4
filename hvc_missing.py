import os
import glob
import re
import numpy as np

def get_missing_pointings(data_dir='HVC_new_data'):
    L_MIN, L_MAX = 90, 180
    B_MIN, B_MAX = 20, 60
    STEP = 2
    TOLERANCE = 0.5  # degrees of tolerance for matching due to messy spacing

    # 1. Generate the theoretical grid
    theoretical_pointings = []
    # Loop over galactic latitude (b)
    for b in np.arange(B_MIN, B_MAX + STEP, STEP):
        # Calculate l step size: 2 / cos(b)
        l_step = STEP / np.cos(np.radians(b))
        
        # Loop over galactic longitude (l)
        for l in np.arange(L_MIN, L_MAX, l_step):
            theoretical_pointings.append((l, b))
            
    # 2. Parse existing files in the directory
    existing_pointings = []
    # Looks for files matching the pattern HVC_l..._b...npz
    file_pattern = os.path.join(data_dir, 'HVC_l*_b*.npz')
    
    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)
        # Extract the l and b floating numbers from the filename using regex
        match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
        if match:
            e_l = float(match.group(1))
            e_b = float(match.group(2))
            existing_pointings.append((e_l, e_b))
            
    # 3. Find the missing pointings
    missing_pointings = []
    
    for t_l, t_b in theoretical_pointings:
        found = False
        for e_l, e_b in existing_pointings:
            # Check if this theoretical point is close to an existing file
            # using the Pythagorean distance in the l-b plane
            distance = np.sqrt(((t_l - e_l) * np.cos(np.radians(t_b)))**2 + (t_b - e_b)**2)
            
            if distance <= TOLERANCE:
                found = True
                break
                
        if not found:
            # If no existing file was close enough, add to missing list
            missing_pointings.append((t_l, t_b))
            
    print(f"Found {len(existing_pointings)} completed pointings.")
    print(f"Identified {len(missing_pointings)} missing pointings.")
    
    return missing_pointings

# If you want to test this script directly:
if __name__ == "__main__":
    missing = get_missing_pointings()
    print("First 5 missing pointings:", missing[:5])