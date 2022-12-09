import librosa
import tensorflow
import numpy

class MusicAnalyzer(tensorflow.keras.Sequential):
    def __init__(self, nclasses, input_shape=()):
        super(MusicAnalyzer, self).__init__()

        self.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))#, input_shape=input_shape))
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

def load_mfccs_and_reshape(path, n_seconds):
    mfccs = []
    y, sr = librosa.load(path)
    while len(y) > sr * n_seconds:
        mfcc = librosa.feature.mfcc(y=y[:sr * n_seconds], sr=sr)
        mfcc = numpy.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1], 1))
        mfccs.append(mfcc)
        y = y[sr * n_seconds:]
        del mfcc
    return mfccs

if __name__ == '__main__':
    import os

    genres = ['disco', 'jazz', 'rock', 'pop', 'blues', 'reggae', 'hiphop', 'metal', 'classical', 'country']
    sec = 30
    epochs = 30

    print(genres)

    def train_model(sec, epochs, genres):
        ma = MusicAnalyzer(len(genres))
        ma.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        for i, genre in enumerate(genres):
            print("Genre: " + genre)
            for file in os.listdir(genre):
                for ext in ma.valid_extensions:
                    if file.endswith(ext):
                        print(f'Loading {file}...')
                        mfccs = load_mfccs_and_reshape(os.path.join(genre, file), sec)
                        print(f'Loaded {len(mfccs)} samples')
                        for mfcc in mfccs:
                            ma.fit(mfcc, numpy.array([i]), epochs=epochs)
                        break
        ma.save('model2.h5')

    def predict(ma, files, sec):
        def print_genre(result, genres):
            print(f'Raw result: {result}')
            print(f'Raw genres: {genres}')
            for i, genre in enumerate(genres):
                print(f'{genre}: {result[0][i] * 100}%')

        print("Results: ")
        for song in files:
            for i, mfcc in enumerate(load_mfccs_and_reshape(song, sec)):
                result = ma.predict(mfcc)
                print(f'File: {song} part {i}')
                print_genre(result, genres)
                print("====================================")
    print(result.summary())
    predict(result, ['./touhou.ogg'], sec)
