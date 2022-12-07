import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa as rosa

class MusicAnalyzerException(Exception):
    def __init__(self, message):
        super().__init__(message)

class MusicAnalyzer:

    def __init__(self):
        self.dataset = {}
        self.model = None

    def add_genre(self, genre):
        if self.dataset.get(genre) is None:
            self.dataset[genre] = []
        return self

    def add_melspectogram(self, genre, specto):
        if self.dataset.get(genre) is None:
            self.add_genre(genre)
        self.dataset[genre].append(specto)
        return self

    def add_melspectograms(self, genre, spectos):
        if self.dataset.get(genre) is None:
            self.add_genre(genre)
        self.dataset[genre].extend(spectos)
        return self

    def build_model(self, layers=[]):
        self.model = tf.keras.models.Sequential(layers)
        return self

    def add_layer(self, layer):
        self.model.add(layer)
        return self

    def compile_model(self,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            loss_weights=None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            **kwargs):

        self.model.compile(
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            optimizer=optimizer,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs)
        return self

    def train(self, epochs=1000, steps_per_epoch=100, validation_split=0.2):
        if self.model is None:
            raise MusicAnalyzerException("Model is not built yet")
        if len(self.dataset) == 0:
            raise MusicAnalyzerException("Dataset is empty")
        for genre in self.dataset:
            self.dataset[genre] = np.array(self.dataset[genre])
        print(self.dataset)
        self.model.fit(
            self.dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_split=validation_split)
        return self

    def save(self, path):
        if self.model is None:
            raise MusicAnalyzerException("Model is not built yet")
        self.model.save(path)
        return self

def load_music_and_extract_random_samples(path, sample_count=10, sample_length=1000):
    samples = []
    music, sr = rosa.load(path)

    for i in range(sample_count):
        start = np.random.randint(0, len(music) - sample_length)
        samples.append(music[start:start+sample_length])
    return samples

def music_samples_to_melspectrogram(samples):
    melspectrogram = []
    for sample in samples:
        melspectrogram.append(rosa.feature.melspectrogram(sample))
    return melspectrogram

def load_musics_and_extract_random_samples(paths, sample_count=10, sample_length=1000):
    musics = []
    for path in paths:
        musics.extend(load_music_and_extract_random_samples(path, sample_count, sample_length))
    return musics

msa = MusicAnalyzer()
msa.add_genre("rock").add_melspectograms("rock",
        music_samples_to_melspectrogram(
            load_musics_and_extract_random_samples([
                "rock/rock1.ogg",
                "rock/rock2.ogg"
            ])
        )
    )
"""
msa.add_genre("pop").add_musics("pop",
        music_samples_to_melspectrogram(
            load_musics_and_extract_random_samples([
                "pop/pop1.ogg",
                "pop/pop2.ogg"
            ])
        ))
"""

msa.build_model([
        #tf.keras.layers.Flatten(input_shape=(128, 128)),
        #tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
msa.compile_model()
msa.train()
msa.save("model.h5")

msa.model.predict(np.array([
        rosa.feature.melspectrogram(rosa.load("rock/rock2.ogg")[0]),
        rosa.feature.melspectrogram(rosa.load("pop/pop2.ogg")[0]),
    ]))
