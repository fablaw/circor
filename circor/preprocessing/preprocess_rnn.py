import numpy as np
import librosa
import librosa.display
import glob
import time
from google.cloud import storage
from circor.parameters.params import BUCKET_NAME, PROJECT
import os

def get_max_length():

    '''Find maximum audio length among all recordings in training_data
    Output used to pad all data at the same length : the largest one'''

    audio_length = list()

    for file in glob.glob('../raw_data/training_data/*.wav')[:2]:
        sig, srate = librosa.load(file, sr = None)
        audio_length.append(sig.shape[0])

    max_length = max(audio_length)

    return max_length

def wav_to_1D_padded(wave_path, wanted_length= 90000, save=False,sr = None):

    '''
    Converting a wav file to a padded ndarray and optional saving.
    Padding is done according to maximum length available in the dataset
     pad_width = (O, N) allows to pad only on the right end of the sequence."""

    '''

    # wav to np.array
    sig, srate = librosa.load(wave_path, sr = sr)
    # add padding
    sig = np.pad(sig,pad_width = (0, wanted_length - sig.shape[0]), mode = 'constant', constant_values = -10)

    if save:

        timestamp = time.strftime('%d_%H_%M') #record day, hours and minute of savings
        np.save(wave_path, sig)
        blob_path = wave_path.split('/')[-1].split('.')[0] #define name of processed_file

        storage_client = storage.Client(project=PROJECT)
        bucket = storage_client.get_bucket(BUCKET_NAME)

        blob = bucket.blob(f"processed_data_1D/{timestamp}/{blob_path}.npy")
        blob.upload_from_filename(wave_path)
        os.remove(wave_path)



    return sig
