import sys

required_libraries = {
    "torch": "PyTorch (Deep Learning framework)",
    "torchvision": "TorchVision (Backbones and transforms)",
    "cv2": "OpenCV (Image and video processing)",
    "mediapipe": "MediaPipe (3D Face Mesh landmarks)",
    "PIL": "Pillow (Image format loading)",
    "numpy": "NumPy (Matrix and array computations)"
}

missing_libraries = []

print("=== CHECKING DEV ENVIRONMENT FOR DEPTH-FAS ===\n")

for lib, desc in required_libraries.items():
    try:
        __import__(lib)
        print(f"[OK] Library '{lib}' is installed. ({desc})")
    except ImportError:
        print(f"[MISSING] Library '{lib}' is NOT installed! ({desc})")
        missing_libraries.append(lib)

print("\n----------------------------------------------")
if missing_libraries:
    print("Your environment is not ready yet. Please run the following command in terminal to install:")
    
    pip_names = []
    for lib in missing_libraries:
        if lib == "cv2":
            pip_names.append("opencv-python")
        elif lib == "PIL":
            pip_names.append("pillow")
        elif lib == "torchvision":
            pass
        elif lib == "torch":
            pip_names.append("torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            pip_names.append(lib.lower())
            
    pip_command = f"pip install {' '.join(set(pip_names))}"
    print(f"\n>>> {pip_command}\n")
    print("(*) Note: If your machine does not have an NVIDIA GPU (CUDA), you can install the CPU version:")
    print("    pip install torch torchvision")
else:
    print("Congratulations! Your dev environment is 100% READY!")
    print("Now you just need to download CASIA-FASD, and run prepare_dataset.py.")
print("----------------------------------------------")

