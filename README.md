<p align="center">
  <img src="https://github.com/HermelaDev/Cats-vs-Dogs_Image_Classification_DL_CNN/blob/main/DOG_vs_CAT.png?raw=true" alt="DOG vs CAT" width="400"/>
</p>

# Cats-vs-Dogs-Image-Classification-CNN-Deep-Learning-Project

This project is a binary image classification model that distinguishes between images of **dogs** and **cats** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## 🧠 Model Summary

- Built using `TensorFlow` and `Keras`
- Trained on labeled images of dogs and cats
- Preprocessing includes normalization and resizing to 128x128
- Model output: binary prediction (0 = Cat, 1 = Dog)

---

## 📁 Project Structure

├── train/ # Training images (referenced by 'Image' and 'Label' in CSV)

├── test/ # Validation/test images

├── model.h5 # Saved trained model

├── wuru.jpeg # Example image used for prediction

├── dog-vs-cat.ipynb # Jupyter Notebook containing training and prediction code

└── README.md # Project documentation (this file)

---

## 🔧 Requirements

- Python 3.7+
- TensorFlow >= 2.0
- NumPy
- Matplotlib
- pandas
- PIL (Pillow)

Install dependencies:

```
pip install tensorflow numpy matplotlib pandas pillow
```

🚀 How to Use
Train the Model

If training from scratch:

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data preparation code here...

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_iterator, epochs=10, validation_data=val_iterator)
model.save("model.h5")
```

2. Load and Predict

```
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import numpy as np
import matplotlib.pyplot as plt

model = load_model("model.h5")
image_path = "/content/wuru.jpeg"
img = load_img(image_path, target_size=(128, 128))
img = np.array(img) / 255.0
img = img.reshape(1, 128, 128, 3)

# Show image
plt.imshow(img.reshape(128, 128, 3))
plt.title("Input Image")
plt.axis('off')
plt.show()

# Predict
pred = model.predict(img)
label = 'Dog' if pred[0] > 0.5 else 'Cat'
print(f"Predicted Label: {label}")
```

📷 Sample Output
![Sample Prediction]("C:\Users\Admin\Documents\GitHub\Cats-vs-Dogs_Image_Classification_DL_CNN\sample.png")

📜 License

MIT License

👩‍💻 Author

Hermela Seltanu Gizaw

