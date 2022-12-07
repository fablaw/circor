import numpy as np
import librosa
import librosa.display
import glob
import time
from google.cloud import storage
import os
from preprocessing.wav_to_1D import download_bucket_objects


project=os.environ.get("PROJECT")
bucket_name=os.environ.get("BUCKET_NAME")

def wav_to_1D(wave = None, save=True ,sr = None, timestamp = time.strftime('%d_%H_%M') ):

    '''
    Converting a wav file to a padded ndarray and optional saving.
    Padding is done according to maximum length available in the dataset
     pad_width = (O, N) allows to pad only on the right end of the sequence."""

    '''
    #retrieving data from google cloud
    local_folder=download_bucket_objects(bucket_name=bucket_name,
                            blob_path= 'audio_raw',
                            local_path = f'circor/raw_data',
                            timestamp=timestamp)
    print(local_folder)
    #retrieving data saved locally
    for wave_file in glob.glob(os.path.join(local_folder,'/{timestamp}/audio_raw/*.wav')):
        sig, srate = librosa.load(wave_file, sr = sr)

    if save:
        new_folder='../processed_data/'
        new_path = new_folder + wave_file.rsplit('.wav')[0].split('/')[-1] + '.npy'
        np.save(new_path, sig)

    return new_folder

if __name__ == '__main__':
    wav_to_1D()
