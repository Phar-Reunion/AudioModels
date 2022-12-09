import librosa
import tensorflow
import numpy

class MusicAnalyzer(tensorflow.keras.Model):
    def __init__(self):
        super(MusicAnalyzer, self).__init__()
        self.conv1 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tensorflow.keras.layers.Flatten()
        self.d1 = tensorflow.keras.layers.Dense(128, activation='relu')
        self.d2 = tensorflow.keras.layers.Dense(10, activation='softmax')
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

    ma = MusicAnalyzer()
    ma.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    genres = ['rock', 'pop']

    for i, genre in enumerate(genres):
        genresnp = numpy.array([0] * len(genres))
        genresnp[i] = 1
        for file in os.listdir(genre):
            for ext in ma.valid_extensions:
                if file.endswith(ext):
                    mfcc = load_mfcc_and_reshape(os.path.join(genre, file), 30)
                    ma.fit(mfcc, numpy.array([i]), epochs=1)
    print(ma.predict(load_mfcc_and_reshape('rock/rock1.ogg', 30)))
    print(ma.predict(load_mfcc_and_reshape('pop/pop2.ogg', 30)))
