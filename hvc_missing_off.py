import os
import glob
import re
import numpy as np

def get_missing_pointings(source_dir='HVC_new_data', dest_dir='HVC_off2_data'):
    """Loads targets from source_dir and returns only those missing from dest_dir."""
    TOLERANCE = 0.5 

    # 1. Get the list of targets we want to observe (from HVC_new_data)
    desired_pointings = []
    source_pattern = os.path.join(source_dir, 'HVC_l*_b*.npz')
    
    if os.path.exists(source_dir):
        for filepath in glob.glob(source_pattern):
            filename = os.path.basename(filepath)
            match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
            if match:
                desired_pointings.append((float(match.group(1)), float(match.group(2))))
    else:
        print(f"Warning: Source directory '{source_dir}' not found.")
        return []

    # 2. Get targets we have already observed (from HVC_off2_data)
    completed_pointings = []
    dest_pattern = os.path.join(dest_dir, 'HVC_l*_b*.npz')
    
    if os.path.exists(dest_dir):
        for filepath in glob.glob(dest_pattern):
            filename = os.path.basename(filepath)
            match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
            if match:
                completed_pointings.append((float(match.group(1)), float(match.group(2))))

    # 3. Find which desired pointings are missing from the completed list
    missing_pointings = []
    for d_l, d_b in desired_pointings:
        found = False
        for c_l, c_b in completed_pointings:
            # Check if coordinates match within tolerance
            distance = np.sqrt((d_l - c_l)**2 + (d_b - c_b)**2)
            if distance <= TOLERANCE:
                found = True
                break
        if not found:
            missing_pointings.append((d_l, d_b))
            
    print(f"Found {len(desired_pointings)} target pointings in '{source_dir}'.")
    print(f"Found {len(completed_pointings)} completed pointings in '{dest_dir}'.")
    print(f"Identified {len(missing_pointings)} missing pointings to observe.")
    
    return missing_pointings

if __name__ == "__main__":
    missing = get_missing_pointings()
    # Pretty print the first few to verify
    for p in missing[:10]:
        print(f"l: {p[0]:.1f}, b: {p[1]:.1f}")
