"""
Main application entry point for Wave Function Collapse with CUDA
"""

import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def check_cuda_availability():
    """Check if CUDA is available and working"""
    try:
        import cupy as cp
        # Try to create a simple array on GPU
        test_array = cp.array([1, 2, 3])
        print(f"CUDA available: {cp.cuda.runtime.getDeviceCount()} device(s)")
        return True
    except ImportError:
        print("Warning: CuPy not available. Falling back to CPU-only mode.")
        return False
    except Exception as e:
        print(f"Warning: CUDA error: {e}. Falling back to CPU-only mode.")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'numpy', 'opencv-python', 'matplotlib', 'PIL', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main application entry point"""
    print("Wave Function Collapse with CUDA")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    if not cuda_available:
        print("Note: Running in CPU-only mode. For better performance, install CUDA toolkit and CuPy.")
    
    try:
        # Import and run GUI
        from gui import WaveFunctionCollapseGUI
        
        print("Starting GUI application...")
        app = WaveFunctionCollapseGUI()
        app.run()
        
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
