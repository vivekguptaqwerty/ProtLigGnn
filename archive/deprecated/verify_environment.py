import sys
import traceback

def main():
    print("==================================================")
    print("      ProtLigGNN Environment Verification         ")
    print("==================================================")
    
    passed = True
    
    # 1. Python Version Check
    try:
        print(f"[+] Python Version: {sys.version}")
        if sys.version_info.major == 3 and sys.version_info.minor == 11:
            print("  -> PASS: Python version is 3.11")
        else:
            print("  -> WARNING: Python version is not 3.11")
            passed = False
    except Exception as e:
        print(f"  -> FAIL: Python check: {e}")
        passed = False

    # 2. PyTorch & CUDA Check
    try:
        import torch
        print(f"[+] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  -> CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"  -> GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  -> CUDA Version: {torch.version.cuda}")
            
            # CUDA Tensor Allocation
            x = torch.randn(3, 3, device="cuda")
            y = torch.randn(3, 3, device="cuda")
            z = x @ y
            print(f"  -> CUDA Tensor computation successful: {z.device}")
            
            # GPU Memory Access
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"  -> GPU Memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
            
            # Mixed Precision Context Check
            with torch.cuda.amp.autocast():
                a = torch.randn(2, 2, device="cuda", dtype=torch.float16)
                b = torch.randn(2, 2, device="cuda", dtype=torch.float16)
                c = a @ b
            print(f"  -> AMP/Autocast context validation: PASS")
        else:
            print("  -> FAIL: CUDA is not available in PyTorch")
            passed = False
    except Exception as e:
        print(f"  -> FAIL: PyTorch / CUDA: {e}")
        traceback.print_exc()
        passed = False

    # 3. PyTorch Geometric (PyG) Check
    try:
        import torch_geometric
        print(f"[+] PyTorch Geometric Version: {torch_geometric.__version__}")
        
        # Verify binary extensions imports
        import torch_scatter
        import torch_sparse
        import torch_cluster
        import torch_spline_conv
        
        print("  -> torch-scatter: PASS")
        print("  -> torch-sparse: PASS")
        print("  -> torch-cluster: PASS")
        print("  -> torch-spline-conv: PASS")
        print("  -> PyG Extensions load: PASS")
    except Exception as e:
        print(f"  -> FAIL: PyG / Extensions: {e}")
        passed = False

    # 4. RDKit Check
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("CCO")
        AllChem.Compute2DCoords(mol)
        print(f"[+] RDKit: PASS (mol created with conformer: {mol.GetNumAtoms()} atoms)")
    except Exception as e:
        print(f"  -> FAIL: RDKit: {e}")
        passed = False

    # 5. BioPython Check
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        print("[+] BioPython PDBParser: PASS")
    except Exception as e:
        print(f"  -> FAIL: BioPython: {e}")
        passed = False

    # 6. Scientific Packages (NumPy, SciPy, pandas, scikit-learn, matplotlib)
    try:
        import numpy as np
        import scipy
        import pandas as pd
        import sklearn
        import matplotlib
        
        print(f"[+] NumPy: Version {np.__version__} (PASS)")
        print(f"[+] SciPy: Version {scipy.__version__} (PASS)")
        print(f"[+] pandas: Version {pd.__version__} (PASS)")
        print(f"[+] scikit-learn: Version {sklearn.__version__} (PASS)")
        print(f"[+] matplotlib: Version {matplotlib.__version__} (PASS)")
    except Exception as e:
        print(f"  -> FAIL: Scientific libraries check: {e}")
        passed = False

    print("==================================================")
    if passed:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
