from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO('yolov8l.pt')  
    source = Path(__file__).parent / "test.webp"
    results = model(source, conf=0.25, save=True)   
    for r in results:
        # Each 'r' object contains information about detections for a single image/frame
        boxes = r.boxes           # Bounding boxes
        masks = r.masks           # Segmentation masks (if using a segmentation model)
        keypoints = r.keypoints   # Keypoints (if using a pose estimation model)
        probs = r.probs           # Class probabilities (if using a classification model)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id] 

            print(f"Detected: {class_name} (Confidence: {confidence:.2f}) "
                f"at Bounding Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # The results are typically saved in 'runs/detect/predict/' directory
    print("\nResults saved to 'runs/detect/predict/' (or similar directory based on previous runs).")





if __name__ == "__main__":
    main()