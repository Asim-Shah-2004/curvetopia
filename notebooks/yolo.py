from ultralytics import YOLO

# Define the actual path of the dataset
data_yaml_path = 'data.yaml'  # YAML configuration file for YOLO

# Create a YAML configuration file for YOLOv8
with open(data_yaml_path, 'w') as f:
    f.write("""
train: C:/Users/Asim/OneDrive/Desktop/projects/curvetopia/dataset2  # Update this to the correct path
val: C:/Users/Asim/OneDrive/Desktop/projects/curvetopia/dataset2  # For demonstration, using the same dataset for validation. Consider separating.

nc: 8  # Number of classes
names:
  0: 'line'
  1: 'triangle'
  2: 'square'
  3: 'circle'
  4: 'ellipse'
  5: 'rectangle'
  6: 'star'
  7: 'regular_polygon'
""")

# Initialize and train the model
model = YOLO('yolov8n.yaml')  # Load YOLOv8 model configuration (YOLOv8n for nano model)

# Train the model
results = model.train(
    data=data_yaml_path,  # Path to YAML file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    project='yolov8_project',  # Project directory to save results
    name='geometric_shapes',  # Name of the run
    cache=True  # Cache images for faster training
)

# Print results
print("Training completed. Results:")
print(results)
