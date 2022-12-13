import librosa
import tensorflow
import numpy
import threading
import random

SECONDS_PER_SAMPLES = 20
EPOCHS = 100
THREAD_COUNT = 100
MUSIC_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
SUPPORTED_EXTENSIONS = ['.ogg', '.wav']

class Atomic:

    def __init__(self, v=None):
        self.__mutex = threading.Lock()
        self.__v = v

    def get(self):
        with self.__mutex:
            return int(self.__v)

    def set(self, v):
        with self.__mutex:
            self.__v = int(v)

class MusicAnalyzerContextState:
    MUSIC_ANALYZER_THREAD_AVIABLE = 0
    MUSIC_ANALYZER_THREAD_BUSY = 1
    MUSIC_ANALYZER_THREAD_FINISHED = 2

class MusicAnalyzerThread(threading.Thread):
    def __init__(self, context, parameters, coroutine):
        threading.Thread.__init__(self)
        self.__context = context
        self.__parameters = parameters
        self.__coroutine = coroutine
        self.result = None

    def run(self):
        self.result = self.__coroutine(self.__parameters)
        self.__context.set(MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_FINISHED)

class MusicAnalyzer:

    def __init__(self, max_threads=THREAD_COUNT):
        self.__max_threads = max_threads
        self.__contexts = [Atomic(MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_AVIABLE) for _ in range(max_threads)]
        self.__threads = [None for _ in range(max_threads)]

        self.__thread_pending_lock = threading.Lock()
        self.__thread_pending = 0

    def run(self, parameters_list, coroutine):
        for i, context in enumerate(self.__contexts):     

            if not len(parameters_list):
                break
            if context.get() != MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_AVIABLE:
                continue
            with self.__thread_pending_lock:
                self.__thread_pending += 1
            context.set(MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_BUSY)
            self.__threads[i] = MusicAnalyzerThread(context, parameters_list[0], coroutine)
            self.__threads[i].start()
            parameters_list = parameters_list[1:]
        return parameters_list

    def get_results(self):
        results = []
        for i, context in enumerate(self.__contexts):
            if context.get() != MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_FINISHED:
                continue
            self.__threads[i].join()
            results.append(self.__threads[i].result)
            self.__threads[i] = None
            context.set(MusicAnalyzerContextState.MUSIC_ANALYZER_THREAD_AVIABLE)
            with self.__thread_pending_lock:
                self.__thread_pending -= 1
        return results

    def has_pending_threads(self):
        with self.__thread_pending_lock:
            return self.__thread_pending != 0

def load_mfccs_and_reshape(params):
    path, genre_index = params

    print("Loading: ", params)

    mfccs = []
    y, sr = librosa.load(path)
    while len(y) > sr * SECONDS_PER_SAMPLES:
        mfcc = librosa.feature.mfcc(y=y[:sr * SECONDS_PER_SAMPLES], sr=sr)
        mfcc = numpy.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1], 1))
        mfccs.append(mfcc)
        y = y[sr * SECONDS_PER_SAMPLES:]
        del mfcc
    print("Loaded: ", params)
    return (mfccs, genre_index)

def make_model(genres=MUSIC_GENRES):
    ma = tensorflow.keras.Sequential()
    ma.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))#, input_shape=input_shape))
    ma.add(tensorflow.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    ma.add(tensorflow.keras.layers.BatchNormalization())

    ma.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    ma.add(tensorflow.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    ma.add(tensorflow.keras.layers.BatchNormalization())

    ma.add(tensorflow.keras.layers.Conv2D(32, (2, 2), activation='relu'))
    ma.add(tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    ma.add(tensorflow.keras.layers.BatchNormalization())

    ma.add(tensorflow.keras.layers.Flatten())
    #ma.add(tensorflow.keras.layers.Dense(64, activation='relu'))
    ma.add(tensorflow.keras.layers.Dropout(0.3))

    for _ in range(5):
        ma.add(tensorflow.keras.layers.Dense(len(genres), activation='softmax'))
    ma.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return ma

if __name__ == '__main__':
    import os
    
    def print_genre(result, genres):
        print(f'Raw result: {result}')
        sorted_values = []
        for i, genre in enumerate(genres):
            sorted_values.append((genre, result[0][i]))
        sorted_values.sort(key=lambda x:x[1])
        for genre, value in sorted_values:
            print(f'{genre}: {value * 100}%')

    def predict(model, files):
        averages = []

        for file in files:
            mfccs, _ = load_mfccs_and_reshape((file, -1))
            music_values = []
            for i, mfcc in enumerate(mfccs):
                v = model.predict(mfcc)
                music_values.append(v[0])
                #print_genre(v, MUSIC_GENRES)
            average = [0] * len(MUSIC_GENRES)
            for v in music_values:
                for i in range(len(MUSIC_GENRES)):
                    average[i] += v[i]
            for i in range(len(MUSIC_GENRES)):
                average[i] /= len(MUSIC_GENRES)
            averages.append((file, average))
        for (file, average) in averages:
            print(f'Average: {file}')
            print_genre([average], MUSIC_GENRES)

    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
        try:
            tensorflow.config.set_logical_device_configuration(
                gpus[0],
                [tensorflow.config.LogicalDeviceConfiguration(memory_limit=1024),
                tensorflow.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tensorflow.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            exit(1)

    analyzer = MusicAnalyzer()
    model = make_model()

    files = []
    for i, genre in enumerate(MUSIC_GENRES):
        for file in os.listdir(genre):
            for ext in SUPPORTED_EXTENSIONS:
                if not file.endswith(ext):
                    continue
                files.append((f'{genre}/{file}', i))
                break
    random.shuffle(files) # homogenisation ?

    while len(files):
        files = analyzer.run(files, load_mfccs_and_reshape)
        for (mfccs, genre_index) in analyzer.get_results():
            for mfcc in mfccs:
                model.fit(mfcc, numpy.array([genre_index]), epochs=EPOCHS)

    while analyzer.has_pending_threads():
        for (mfccs, genre_index) in analyzer.get_results():
            for mfcc in mfccs:
                model.fit(mfcc, numpy.array([genre_index]), epochs=EPOCHS)
    model.save("threaded_test.h5")
 
    predict(tensorflow.keras.models.load_model("threaded_test.h5"), ['acdc.wav', 'amstrong.ogg', 'one-love.ogg'])
    exit(0)
