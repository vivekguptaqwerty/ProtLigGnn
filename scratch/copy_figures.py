import os
import shutil
from pathlib import Path

def main():
    workspace_dir = Path(r"d:\ProtLigGnn")
    out_dir = Path(r"C:\Users\Harshhhh\.gemini\antigravity\brain\03c21182-9e56-4764-a6fd-83660d6d0fc1")
    
    print("Scanning workspace for PNG figures...")
    
    # Scan experiments/ for PNGs
    for root, dirs, files in os.walk(workspace_dir / "experiments"):
        for f in files:
            if f.endswith(".png"):
                src_path = Path(root) / f
                dest_path = out_dir / f
                # Avoid overwriting with older files if already exists
                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {f} to artifacts")
                    
    # Scan root directory for PNGs
    for f in os.listdir(workspace_dir):
        if f.endswith(".png"):
            src_path = workspace_dir / f
            dest_path = out_dir / f
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"Copied {f} to artifacts")
                
    print("Scanning complete.")

if __name__ == "__main__":
    main()
