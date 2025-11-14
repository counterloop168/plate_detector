import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA Version: N/A")
    print("GPU Device: N/A")
    print("\nℹ️  GPU acceleration is NOT enabled.")
    print("   Install CUDA-enabled PyTorch for 5-10x faster processing:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
