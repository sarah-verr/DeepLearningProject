import json, os
from typing import List, Dict, Tuple
import subprocess, sys, os, time

def run_stage(script: str, desc: str):
    print(f"\n{'='*60}")
    print(f"▶️  Stage: {desc}")
    print(f"{'='*60}")
    start = time.time()
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        sys.exit(1)
    end = time.time()
    print(f"Finished {desc} in {end-start:.1f}s")



def main():
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        ("img-gen.py", "Generate base geometric scenes"),
        ("ann-gen.py", "Add pairwise relations, captions, and QA"),
        ("aug-gen.py", "Apply geometric augmentations and remap relations"),
    ]
    for fname, desc in scripts:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {fname} (file not found).")
            continue
        run_stage(path, desc)
    

if __name__ == "__main__":
    main()
