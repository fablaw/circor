import numpy as np
import librosa
import librosa.display

def rgba_new(X_pred):

    x, sr=librosa.load(X_pred)

    D = librosa.stft(x[0:50000], n_fft=446, hop_length=148)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    spectrogram = librosa.display.specshow(S_db, y_axis="log", sr=sr, hop_length=1024, x_axis="time")

    rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(S_db.shape))

    rgba=rgbas[:,:,0:3]

    X_new=np.expand_dims(rgba, axis=0)

    return X_new
