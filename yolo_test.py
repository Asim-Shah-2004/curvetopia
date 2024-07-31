from ultralytics import YOLO
import cv2

# Load the model
model = YOLO(r'model/yolov8_project/geometric_shapes_cpu6/weights/best.pt')  # or 'path/to/your/best.pt'

# Load an image
img = cv2.imread(r'C:\Users\Asim\OneDrive\Desktop\projects\curvetopia\testData\image.png')

# Run inference
results = model(img)

# Print results
print(results)