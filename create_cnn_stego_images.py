import os
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

# Create folders
os.makedirs("real_dataset/cover", exist_ok=True)
os.makedirs("real_dataset/stego", exist_ok=True)

# Load MNIST (handwritten digits)
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train[:500]  # Use 500 images for this test

# Save cover images
for i, img in enumerate(x_train):
    cv2.imwrite(f"real_dataset/cover/cover_{i}.png", img)

# Embed message into LSB
def embed_lsb(image, message="HiddenMessage"):
    bits = ''.join(format(ord(c), '08b') for c in message)
    flat = image.flatten()
    for i in range(min(len(bits), len(flat))):
        flat[i] = np.uint8((flat[i] & 0b11111110) | int(bits[i]))
    return flat.reshape(image.shape)


# Save stego images
for i, img in enumerate(x_train):
    stego_img = embed_lsb(img.copy())
    cv2.imwrite(f"real_dataset/stego/stego_{i}.png", stego_img)
