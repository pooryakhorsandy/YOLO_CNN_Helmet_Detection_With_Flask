"""
In this segment, our objective is to generate CNN weights for our helmet detection program.
 Feel free to utilize your own image to develop these weights.
"""


# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Define constants
TRAIN_DIR = r'\output_folder\train'
TEST_DIR = r'\output_folder\test'
IMG_SIZE = (224, 224)

# Create data augmentation generator for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create data augmentation generator for testing data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the VGG16 model (without the top dense layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new top dense layers for classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define batch size for data generators
batch_size = 32

# Create data generators for training and testing data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode='binary'
)

# Train the model
epochs = 20
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Save the trained model
model.save('final_cnn_helmet_detection_weight.h5')
