import librosa
import librosa.display
import numpy as np
from circor.parameters.params import BUCKET_NAME, PROJECT
import os
from google.cloud import storage
import time

#Transform audio-recordings in 2D array for CNN input

def wav_to_2D_HD(wave_path,hop_length=1024, save=False, timestamp = time.strftime('%d_%H_%M')):
    '''Input = .wav file format ie. and audio recording
        Returns a np array RGBA'''

    wav_audio, fs=librosa.load(wave_path, sr=None)
    scaled_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(wav_audio, hop_length=hop_length)), ref=np.max)

    spectrogram = librosa.display.specshow(scaled_spectrogram, y_axis="linear", sr=fs, hop_length=hop_length, x_axis="time")

    rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(scaled_spectrogram.shape))

    if save:
        new_path = f"{'/'.join(wave_path.split('.')[0].split('/')[:-1])}/test_numpy/{wave_path.split('.')[0].split('/')[-1]}.npy"
        np.save(new_path, rgbas)
        blob_path = wave_path.split('/')[-1].split('.')[0]
        """Uploads a file to the bucket."""
        storage_client = storage.Client(project=PROJECT)
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(f"processed_data_2D/{timestamp}_{blob_path}.npy")
        blob.upload_from_filename(new_path)
        os.remove(wave_path)

    return rgbas




def wav_to_2D_compressed(wave_path,hop_length=1024, save=False, timestamp = time.strftime('%d_%H_%M')):

    wav_audio, fs=librosa.load(wave_path, sr=None)
    scaled_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(wav_audio, hop_length=hop_length)), ref=np.max)

    spectrogram = librosa.display.specshow(scaled_spectrogram, y_axis="linear", sr=fs, hop_length=hop_length, x_axis="time")

    rgbas = spectrogram.to_rgba(spectrogram.get_array().reshape(scaled_spectrogram.shape))*255
    rgbas = rgbas.astype('uint8')

    if save:
        new_path = f"{'/'.join(wave_path.split('.')[0].split('/')[:-1])}/test_numpy/{wave_path.split('.')[0].split('/')[-1]}.npy"
        np.save(new_path, rgbas)
        blob_path = wave_path.split('/')[-1].split('.')[0]
        """Uploads a file to the bucket."""
        storage_client = storage.Client(project=PROJECT)
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(f"processed_data_2D/{timestamp}/{blob_path}.npy")
        blob.upload_from_filename(new_path)
        os.remove(wave_path)

    return rgbas
