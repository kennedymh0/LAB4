import os
import glob
import re
import numpy as np

def get_missing_pointings(data_dir='HVC_off_data'):
    L_MIN, L_MAX = 60, 180
    B_MIN, B_MAX = 20, 60
    L_STEP = 20  # Based on your example (60, 80...)
    B_STEP = 2   # Based on your example (20, 22...)
    TOLERANCE = 0.5 

    # 1. Generate the theoretical grid
    theoretical_pointings = []
    
    # Outer loop is Latitude (b) to group all 'l's of the same 'b' together
    # Or swap these if you want all 'b's for a single 'l' first.
    # Based on your example: (60,20), (80,20) -> l changes, b stays same.
    for b in np.arange(B_MIN, B_MAX + B_STEP, B_STEP):
        for l in np.arange(L_MIN, L_MAX + L_STEP, L_STEP):
            theoretical_pointings.append((l, b))
            
    # 2. Parse existing files in the directory
    existing_pointings = []
    file_pattern = os.path.join(data_dir, 'HVC_l*_b*.npz')
    
    if os.path.exists(data_dir):
        for filepath in glob.glob(file_pattern):
            filename = os.path.basename(filepath)
            match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
            if match:
                existing_pointings.append((float(match.group(1)), float(match.group(2))))
            
    # 3. Find the missing pointings
    missing_pointings = []
    for t_l, t_b in theoretical_pointings:
        found = False
        for e_l, e_b in existing_pointings:
            # Simple Euclidean distance since we are looking for specific grid matches
            distance = np.sqrt((t_l - e_l)**2 + (t_b - e_b)**2)
            if distance <= TOLERANCE:
                found = True
                break
        if not found:
            missing_pointings.append((t_l, t_b))
            
    print(f"Found {len(existing_pointings)} completed pointings.")
    print(f"Identified {len(missing_pointings)} missing pointings.")
    
    return missing_pointings

if __name__ == "__main__":
    missing = get_missing_pointings()
    # Pretty print the first few to verify order
    for p in missing[:10]:
        print(f"l: {p[0]:.1f}, b: {p[1]:.1f}")
