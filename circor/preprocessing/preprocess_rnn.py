import numpy as np
import librosa
import librosa.display
import glob
import time
from google.cloud import storage
from circor.parameters.params import BUCKET_NAME, PROJECT
import os

storage_client = storage.Client(project=PROJECT)
bucket = storage_client.get_bucket(BUCKET_NAME)


def wav_to_1D(wave, save=False, sr = None, timestamp = time.strftime('%d_%H_%M') ):

    '''
    Converting a wav file to a padded ndarray and optional saving.
    Padding is done according to maximum length available in the dataset
     pad_width = (O, N) allows to pad only on the right end of the sequence."""

    '''
    for wave_path in wave:
    # wav to np.array
        sig, srate = librosa.load(wave_path, sr = sr)

        if save:
            new_path = f"{'/'.join(wave_path.split('.')[0].split('/')[:-1])}/test_numpy/{wave_path.split('.')[0].split('/')[-1]}.npy"
            np.save(new_path, sig)
            blob_path = wave_path.split('/')[-1].split('.')[0] #define name of processed wave file eg: '2530'


        #locate file in dedicated folder named after timestamp
            blob = bucket.blob(f"raw_1D/{timestamp}/{blob_path}.npy")
            blob.upload_from_filename(new_path)
            os.remove(wave_path)

        return sig
