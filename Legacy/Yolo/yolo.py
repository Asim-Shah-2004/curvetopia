from ultralytics import YOLO

# Define the actual path of the dataset
dataset_path = './dataset3'
data_yaml_path = 'data.yaml'  # YAML configuration file for YOLO

# Verify dataset path exists


# Create a YAML configuration file for YOLOv8
with open(data_yaml_path, 'w') as f:
    f.write(f"""
train: {dataset_path}  # Update this to the correct path
val: {dataset_path}  # For demonstration, using the same dataset for validation. Consider separating.

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
    epochs=10,  # Number of training epochs, start with fewer epochs
    imgsz=320,  # Smaller image size to reduce computational load
    batch=2,    # Reduce batch size to manage memory usage
    device='cpu',  # Explicitly specify CPU
    project='yolov8_project',  # Project directory to save results
    name='geometric_shapes_cpu',  # Name of the run
    cache=False,  # Disable cache to save RAM
    verbose=True,  # Enable verbose to get more detailed logs
    workers=2  # Number of workers for data loading
)

# Print results
print(f'Average Epoch Duration: {results.times.mean()} seconds')
print(f'Estimated Total Training Time for 10 epochs: {results.times.mean() * 10} seconds')
print("Training completed. Results:")
print(results)
