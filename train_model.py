import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import emnist # Library dataset
import matplotlib.pyplot as plt

print("Sedang mendownload dan memuat dataset EMNIST...")
# Kita gunakan split 'balanced' agar jumlah sampel huruf dan angka seimbang
# Ini akan mendownload dataset secara otomatis
images_train, labels_train = emnist.extract_training_samples('balanced')
images_test, labels_test = emnist.extract_test_samples('balanced')

# Mapping label (EMNIST balanced memiliki 47 kelas: 0-9, A-Z, a-z tertentu)
# Note: EMNIST menggabungkan huruf besar/kecil yang mirip (seperti C dan c), total 47 kelas.
num_classes = 47

# Preprocessing Data
# Normalisasi pixel dari 0-255 menjadi 0-1
images_train = images_train.astype('float32') / 255
images_test = images_test.astype('float32') / 255

# Reshape agar sesuai input CNN (28x28 pixel, 1 channel grayscale)
images_train = images_train.reshape(images_train.shape[0], 28, 28, 1)
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)

# One-hot encoding untuk label
y_train = to_categorical(labels_train, num_classes)
y_test = to_categorical(labels_test, num_classes)

print("Membangun Arsitektur CNN...")
model = Sequential([
    # Layer Konvolusi 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    # Layer Konvolusi 2
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flattening (Mengubah matriks gambar menjadi vektor lurus)
    Flatten(),

    # Fully Connected Layer
    Dense(128, activation='relu'),
    Dropout(0.5), # Mencegah overfitting

    # Output Layer
    Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
print("Mulai Training (ini mungkin memakan waktu)...")
history = model.fit(images_train, y_train, batch_size=128, epochs=10, validation_data=(images_test, y_test))

# Simpan Model
model.save('handwriting_cnn_model.h5')
print("Model berhasil disimpan sebagai 'handwriting_cnn_model.h5'")

# Label Mapping (Kamus untuk menerjemahkan angka hasil prediksi ke Huruf)
# EMNIST Balanced mapping
label_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
print("Mapping Label Selesai.")