import os
import sys
from pathlib import Path

def check_dataset():
    print("Checking dataset structure...")
    
    base_path = Path("../DIV2KDataset")
    
    # Check required folders
    required = [
        ("DIV2K_train_HR", "HR training images"),
        ("DIV2K_train_LR_bicubic/X2", "LR training images (2x)"),
        ("DIV2K_valid_HR", "HR validation images"),
        ("DIV2K_valid_LR_bicubic/X2", "LR validation images (2x)")
    ]
    
    all_ok = True
    
    for folder, description in required:
        folder_path = base_path / folder
        if folder_path.exists():
            png_files = list(folder_path.glob("*.png"))
            if png_files:
                print(f"✓ {description}: {len(png_files)} images")
                
                # Check image dimensions
                if png_files:
                    try:
                        import cv2
                        img = cv2.imread(str(png_files[0]))
                        if img is not None:
                            print(f"  Sample size: {img.shape}")
                    except:
                        pass
            else:
                print(f"✗ {description}: No PNG files found")
                all_ok = False
        else:
            print(f"✗ {description}: Folder not found")
            all_ok = False
    
    return all_ok

def check_imports():
    print("\nChecking Python imports...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ NumPy: {np.__version__}")
        print(f"✓ OpenCV: {cv2.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
        else:
            print("ℹ CUDA not available, using CPU")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_file_structure():
    print("\nChecking project file structure...")
    
    required_files = [
        "main.py",
        "data_preprocessing.py",
        "srcnn_model.py",
        "edsr_model.py"
    ]
    
    all_ok = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ Missing: {file}")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("Project Setup Validation")
    print("="*60)
    
    tests = [
        ("Dataset Structure", check_dataset),
        ("Python Imports", check_imports),
        ("File Structure", check_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Validation Summary:")
    print("-" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All checks passed! You can run the project.")
        print("\nTo run in quick test mode:")
        print("  python main.py --scale 2 --quick")
        print("\nFor full training:")
        print("  python main.py --scale 2 --epochs 10 --batch 8")
    else:
        print("Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)