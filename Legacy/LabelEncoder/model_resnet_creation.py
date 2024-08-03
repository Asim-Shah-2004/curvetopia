import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Ensure TensorFlow uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


def preprocess_grayscale_to_rgb(img):
    img = img_to_array(img)  # Convert to array
    img = np.stack([img[:, :, 0]] * 3, axis=-1)  # Stack the grayscale channel to make RGB
    return img


def load_labels_from_directory(base_dir):
    """
    Load labels from the specified directory.

    Parameters:
    - base_dir (str): The base directory containing subdirectories for each shape type.

    Returns:
    - labels (list): List of shape labels corresponding to the images.
    """
    labels = []
    shape_types = ['line', 'triangle', 'square', 'circle', 'rectangle', 'star', 'regular_polygon']
    for shape in shape_types:
        shape_dir = os.path.join(base_dir, shape)
        if os.path.isdir(shape_dir):
            for filename in os.listdir(shape_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    labels.append(shape)
    return labels


# Directory path
dataset_directory = '../dataset'

# Load labels and prepare data generators
labels = load_labels_from_directory(dataset_directory)
label_encoder = LabelEncoder()
encoded_labels = to_categorical(label_encoder.fit_transform(labels))

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_grayscale_to_rgb,
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the ResNet50 model with custom input
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Clear previous TensorFlow sessions
K.clear_session()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(val_gen)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('../models/shapes_model_resnet.keras')
