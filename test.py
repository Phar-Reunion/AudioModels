import librosa
import tensorflow
import numpy

class MusicAnalyzer(tensorflow.keras.Sequential):
    def __init__(self, nclasses, input_shape):
        super(MusicAnalyzer, self).__init__()

        self.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.add(tensorflow.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.add(tensorflow.keras.layers.BatchNormalization())

        self.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.add(tensorflow.keras.layers.BatchNormalization())

        self.add(tensorflow.keras.layers.Conv2D(32, (2, 2), activation='relu'))
        self.add(tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self.add(tensorflow.keras.layers.BatchNormalization())

        self.add(tensorflow.keras.layers.Flatten())
        self.add(tensorflow.keras.layers.Dense(64, activation='relu'))
        self.add(tensorflow.keras.layers.Dropout(0.3))

        self.add(tensorflow.keras.layers.Dense(nclasses, activation='softmax'))

        self.valid_extensions = ['.ogg', '.wav']

def load_mfcc(path, n_seconds=-1):
    y, sr = librosa.load(path)
    if n_seconds > 0:
        y = y[:sr * n_seconds]
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc, sr

def load_mfcc_and_reshape(path, n_seconds=-1):
    mfcc, sr = load_mfcc(path, n_seconds)
    mfcc = numpy.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1], 1))
    return mfcc

if __name__ == '__main__':
    import os

    genres = ['rock', 'pop', 'shatta', 'zouk', 'classic']
    sec = 60
    epochs = 30

    ma = MusicAnalyzer(len(genres))
    ma.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for i, genre in enumerate(genres):
        print("Genre: " + genre)
        for file in os.listdir(genre):
            for ext in ma.valid_extensions:
                if file.endswith(ext):
                    print(f'Loading {file}...')
                    mfcc = load_mfcc_and_reshape(os.path.join(genre, file), sec)
                    ma.fit(mfcc, numpy.array([i]), epochs=epochs) # numpy.array([i]) is the label for the genre but it does not work as the always returned label is the last one
                    break

    def print_genre(result, genres):
        print(result)

    #ma.save('model.h5')

    print("Results: ")
    for song in ["../skylar.wav"]:
        result = ma.predict(load_mfcc_and_reshape(song, sec))
        print(f'File: {song}')
        print_genre(result, genres)
        print("====================================")
