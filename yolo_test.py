from ultralytics import YOLO
import cv2

# Load the model
model = YOLO(r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\model\content\yolov8_project\geometric_shapes2\weights\last.pt')  # or 'path/to/your/best.pt'

# Load an image
img = cv2.imread(r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\dataset2\star\star_1.png')

# Run inference
results = model(img)

# Print results
print(results)