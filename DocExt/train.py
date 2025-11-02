from ultralytics import YOLO
import os

def train_model():
    """
    Trains the YOLOv8 model on the custom dataset.
    """
    # Load a pre-trained YOLOv8 model (e.g., yolov8n.pt for a small, fast model)
    # You can choose yolov8s, yolov8m, etc., for better accuracy at the cost of speed.
    model = YOLO('yolov8n.pt')

    # Train the model
    # This assumes 'data.yaml' is in the same directory or you provide the correct path.
    print("Starting model training...")
    try:
        results = model.train(
            data='data.yaml',
            epochs=50,  # Start with 50 and see how it performs
            imgsz=640,  # Standard image size
            batch=8,
            name='id_card_detector'  # A name for the training run
        )
        print("Training complete.")
        print(f"Model saved to: {results.save_dir}/weights/best.pt")
        print(f"Please move 'best.pt' to this project's main directory and rename it 'best_model.pt'.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure your 'data.yaml' file paths (train, val, test) are correct.")
        print("The paths in data.yaml should be relative to the directory where you run this 'train.py' script, or be absolute paths.")

if __name__ == "__main__":
    # Check for data.yaml
    if not os.path.exists('data.yaml'):
        print("Error: 'data.yaml' not found.")
        print("Please make sure 'data.yaml' is in the same directory as 'train.py'.")
    else:
        train_model()
