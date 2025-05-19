import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Folder structure
base_dir = "real_dataset"
original_dir = os.path.join(base_dir, "originals")
cover_dir = os.path.join(base_dir, "covers")
stego_dir = os.path.join(base_dir, "stegos")

def load_images_from_folder(folder, label, image_size=(64, 64)):
    data, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append(img)
            labels.append(label)
    return data, labels

# Load cover and stego images
original_imgs, original_lbls = load_images_from_folder(original_dir, label=2)
cover_imgs, cover_lbls = load_images_from_folder(cover_dir, label=0)
stego_imgs, stego_lbls = load_images_from_folder(stego_dir, label=1)

# Combine and preprocess
X = np.array(cover_imgs + stego_imgs + original_imgs, dtype=np.float32) / 255.0
y = np.array(cover_lbls + stego_lbls + original_lbls)


X = X.reshape(-1, 64, 64, 1)
y = to_categorical(y, 3)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # for 3 classes: original, cover, stego
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("cnn_model_real_stegoappdb.h5")
print("✅ Model training complete and saved as cnn_model_real_stegoappdb.h5")
