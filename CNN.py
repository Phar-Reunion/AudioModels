import numpy as np
import pandas as pd
import os
import json
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

SOURCE_PATH = 'genres_original'
JSON_PATH   = 'rodata.json'

sr = 22050
TOTAL_SAMPLES = 29 * sr
NUM_SLICES = 10
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)
    
GENRE_DICT = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

def preprocess_data(source_path, json_path):
    mydict = {
        "labels": [],
        "mfcc": []
    }

    count = 0
    for i, genre in enumerate(GENRE_DICT):
        genre_path = f'{SOURCE_PATH}/{genre}'
        for file in os.listdir(genre_path):
            if file in ['.', '..']:
                continue
            print(f'Loading {file} - {count + 1}')
            song, sr = librosa.load(f'{genre_path}/{file}', duration=29)
            for s in range(NUM_SLICES):
                start_sample = SAMPLES_PER_SLICE * s
                end_sample = start_sample + SAMPLES_PER_SLICE
                mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=13)
                mfcc = mfcc.T
                mydict["labels"].append(i)
                mydict["mfcc"].append(mfcc.tolist())
            count += 1
    with open(json_path, 'w') as f:
        json.dump(mydict, f)
    f.close()


def load_data(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
    f.close()
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def prepare_datasets(inputs, targets, split_size):
    # Creating a validation set and a test set.
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=split_size)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs_train, targets_train, test_size=split_size)
    
    # Our CNN model expects 3D input shape.
    inputs_train = inputs_train[..., np.newaxis]
    inputs_val = inputs_val[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]
    
    return inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test


def design_model(input_shape):

    # Let's design the model architecture.
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(len(np.unique(targets)), activation='softmax')
    ])

    return model

def make_prediction(model, X):
    predictions = model.predict(X)
    
    for prediction in predictions:
        genre_max = np.argmax(prediction)
        for i, genre_type in enumerate(GENRE_DICT):
            print(f'GenreType: {genre_type} -> {prediction[i] * 100}%')
        print(f'Genre: np.argmax {GENRE_DICT[genre_max]}')

if __name__ == "__main__":
    preprocess_data(source_path=SOURCE_PATH, json_path=JSON_PATH)
    inputs, targets = load_data(json_path=JSON_PATH)

    Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(inputs, targets, 0.2)
    model = design_model((Xtrain.shape[1], Xtrain.shape[2], 1))

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics = ['acc']
                     )
    model.summary()
    history = model.fit(Xtrain, ytrain,
                        validation_data=(Xval, yval),
                        epochs=30,
                        batch_size=32
                        )

    for file in ['acdc.wav', 'one-love.wav', 'wednesday.wav']:
        song, sr = librosa.load(file, duration=29)
        X = []
        for s in range(NUM_SLICES):
            start_sample = SAMPLES_PER_SLICE * s
            end_sample = start_sample + SAMPLES_PER_SLICE
            mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=13)
            mfcc = mfcc.T
            X.append(mfcc.tolist())
        print(f'Predicting: {file}')
        make_prediction(model, np.array(X))
        print('============================')
