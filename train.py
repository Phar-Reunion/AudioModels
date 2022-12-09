import tensorflow as tf
import numpy as np
import librosa
import os

# Load the audio data and labels for the different genders
audio_data = "../zouk/"
gender_labels = [1]

# Extract features from the audio data using librosa
features = []

for data in os.listdir(audio_data):
    if not data.endswith(".wav"):
        continue
    audio, sr = librosa.load(audio_data + data)
    mfccs = librosa.feature.mfcc(audio, sr=sr)
    features.append(mfccs)
print(features)
# Convert the features and labels to numpy arrays
features = np.array(features)


# Build the model using the TensorFlow Keras API
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim=features.shape(), activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(gender_labels.shape[1], activation='softmax'))

# Compile the model with an optimizer and a loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the audio data
model.fit(features, gender_labels, epochs=20)
