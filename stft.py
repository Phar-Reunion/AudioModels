#!/usr/bin/env python3

import librosa
import numpy as np
import librosa.display as ld
import matplotlib.pyplot as plt

song, sr = librosa.load('../zouk/Chiktay - La pli si Tol-jzwFy6VSg_A.wav')

ft = librosa.stft(song)
S_db = librosa.amplitude_to_db(np.abs(ft), ref=np.max)

fig, ax = plt.subplots()
img = ld.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Now with labeled axes!')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()