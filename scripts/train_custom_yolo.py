"""
Custom YOLO11 Fine-Tuning Script
Adapts the base YOLO11 model to detect vehicles in CADP CCTV footage.
"""
from ultralytics import YOLO
from pathlib import Path

def fine_tune_yolo():
    print("Initialising YOLO11 Nano base model...")
    # Load the pre-trained COCO model (the "smart" base)
    model = YOLO('yolo11n.pt') 
    
    # --- IMPORTANT: Update this path ---
    # Ensure this points to the data.yaml file inside your extracted Roboflow folder
    yaml_path = Path('./data/cadpannotated/data.yaml').absolute()
    
    if not yaml_path.exists():
        print(f"❌ Cannot find YAML file at: {yaml_path}")
        print("Please check the folder name and try again.")
        return

    print(f"✅ Found dataset configuration at: {yaml_path}")
    print("🚀 Starting fine-tuning process. This may take a little while...")
    
    # Train the model
    results = model.train(
        data=str(yaml_path),
        epochs=50,                  # 50 is a great starting point for fine-tuning
        imgsz=640,                  # Standard YOLO resolution
        batch=16,                   # Lower this to 8 if your GPU runs out of VRAM
        device=0,                   # 0 forces it to use your primary NVIDIA GPU
        project='cadp_custom_yolo', # Directory where your new model will be saved
        name='v1_traffic_model',    # Name of this specific training run
        patience=10,                # Stops early if accuracy stops improving for 10 epochs
        augment=True                # Utilises YOLO's built-in data augmentations
    )
    
    print("\n🎉 Fine-tuning complete!")
    print("Your new custom weights are saved at: cadp_custom_yolo/v1_traffic_model/weights/best.pt")

if __name__ == "__main__":
    fine_tune_yolo()