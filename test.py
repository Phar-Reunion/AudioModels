import librosa
import tensorflow
import numpy

class MusicAnalyzer(tensorflow.keras.Model):
    def __init__(self, nclasses):
        super(MusicAnalyzer, self).__init__()
        self.conv1 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tensorflow.keras.layers.Flatten()
        self.d1 = tensorflow.keras.layers.Dense(128, activation='relu')
        self.d2 = tensorflow.keras.layers.Dense(nclasses, activation='softmax')
        self.valid_extensions = ['.ogg', '.wav']

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

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
