import os
import glob
import re

def get_existing_pointings(data_dir='HVC_new_data'):
    """Parses existing .npz files in the specified directory and extracts (l, b) targets."""
    existing_pointings = []
    file_pattern = os.path.join(data_dir, 'HVC_l*_b*.npz')
    
    if os.path.exists(data_dir):
        for filepath in glob.glob(file_pattern):
            filename = os.path.basename(filepath)
            # Extract l and b using regex
            match = re.search(r'HVC_l([0-9.]+)_b([0-9.]+)\.npz', filename)
            if match:
                l_deg = float(match.group(1))
                b_deg = float(match.group(2))
                existing_pointings.append((l_deg, b_deg))
                
    print(f"Found {len(existing_pointings)} existing pointings in '{data_dir}'.")
    return existing_pointings

if __name__ == "__main__":
    existing = get_existing_pointings()
    # Pretty print the first few to verify
    for p in existing[:10]:
        print(f"l: {p[0]:.1f}, b: {p[1]:.1f}")
