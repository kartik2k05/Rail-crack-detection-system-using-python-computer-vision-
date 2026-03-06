from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os

# Load trained model
model = YOLO("runs/detect/train7/weights/best.pt")

# Load all test images
image_list = glob.glob("test/images/*.jpg") + glob.glob("test/images/*.png")

# Create output folder like YOLO runs
save_dir = "runs/heatmaps"
os.makedirs(save_dir, exist_ok=True)

for img_path in image_list:

    print(f"Processing {img_path}")

    results = model(img_path)

    img = cv2.imread(img_path)
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)

    if len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (51,51), 0)

    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = os.path.basename(img_path)
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, overlay)

print(f"\nHeatmaps saved in: {save_dir}")