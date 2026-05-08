import os
import glob
import re
import numpy as np
from collections import defaultdict

def get_missing_pointings(data_dir='HVC_new_data'):
    # --- TARGET BOUNDS ---
    # We set L_MIN to 60 to accommodate your earlier observations!
    L_MIN = 60  
    L_MAX = 180
    B_MIN = 20
    B_MAX = 60
    B_STEP = 2
    # ---------------------
    
    b_tracks = defaultdict(list)
    
    # 1. Read existing files and group longitudes by their latitude track
    file_pattern = os.path.join(data_dir, 'HVC_l*_b*.npz')
    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)
        match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
        if match:
            e_l = float(match.group(1))
            e_b = float(match.group(2))
            
            # Round slightly to group tiny floating point variations together
            b_key = round(e_b, 1) 
            
            # Skip odd numbered b values
            if b_key % 2 != 0:
                continue
                
            b_tracks[b_key].append(e_l)
            
    # 2. Add empty lists for any theoretical latitudes you completely missed
    for b in np.arange(B_MIN, B_MAX + B_STEP, B_STEP):
        b_key = round(float(b), 1)
        if b_key not in b_tracks:
            b_tracks[b_key] = []
            
    missing_pointings = []
    
    # 3. Find the gaps in each track!
    for b, l_list in b_tracks.items():
        # Calculate the proper step size for this specific latitude
        l_step = 2.0 / np.cos(np.radians(b))
        
        # If we have absolutely no data for this latitude, generate the whole track
        if not l_list:
            for l in np.arange(L_MIN, L_MAX + l_step, l_step):
                if l <= L_MAX:
                    missing_pointings.append((round(l, 2), b))
            continue
            
        # Otherwise, sort what we have and look for holes
        start_l = min(L_MIN, min(l_list))
        end_l = max(L_MAX, max(l_list))
        sorted_l = sorted(l_list)
        
        # Artificial anchor if the track starts too late (e.g., started at 120 instead of 60)
        if sorted_l[0] - start_l > 1.5 * l_step:
            sorted_l.insert(0, start_l - l_step)
            
        # Artificial anchor if the track stops too early (e.g., stopped at 150 instead of 180)
        if end_l - sorted_l[-1] > 1.5 * l_step:
            sorted_l.append(end_l + l_step)
            
        # Sweep through the points on this track
        for i in range(len(sorted_l) - 1):
            l_curr = sorted_l[i]
            l_next = sorted_l[i+1]
            
            # If the gap between two recorded points is larger than ~1.5x the step size, it's a hole!
            if l_next - l_curr > 1.5 * l_step:
                
                # Generate points to fill this specific hole
                gap_points = np.arange(l_curr + l_step, l_next - 0.5 * l_step, l_step)
                
                for gp in gap_points:
                    # Only add if it's within your overall target bounds
                    if L_MIN <= gp <= L_MAX:
                        missing_pointings.append((round(gp, 2), b))
                        
    # Sort the final list by latitude, then longitude so the telescope sweeps smoothly
    if not missing_pointings:
        return []

    # Helper function to calculate angular distance, correcting for l-compression
    def angular_distance(p1, p2):
        l1, b1 = p1
        l2, b2 = p2
        avg_b_rad = np.radians((b1 + b2) / 2.0)
        # Scale delta-l by cos(b) to get true angular step size
        dl = (l1 - l2) * np.cos(avg_b_rad)
        db = b1 - b2
        return np.sqrt(dl**2 + db**2)

    unvisited = missing_pointings.copy()
    optimized_pointings = []
    
    # Start at the target with the lowest b, then lowest l
    current_target = min(unvisited, key=lambda x: (x[1], x[0]))
    unvisited.remove(current_target)
    optimized_pointings.append(current_target)
    
    # Keep finding the closest unvisited point until the list is empty
    while unvisited:
        next_target = min(unvisited, key=lambda p: angular_distance(current_target, p))
        unvisited.remove(next_target)
        optimized_pointings.append(next_target)
        current_target = next_target
        
    missing_pointings = optimized_pointings

    print(f"Directory scan complete. Identified {len(missing_pointings)} missing targets.")
    
    return missing_pointings

if __name__ == "__main__":
    missing = get_missing_pointings()
    print("First 10 missing targets:")
    for m in missing[:10]:
        print(m)
