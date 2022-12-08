import numpy as np
import librosa
import librosa.display
import time
from google.cloud import storage
from circor.parameters.params import BUCKET_NAME, PROJECT
import os

def download_bucket_objects(bucket_name, blob_path, local_path, timestamp):
    # blob path is bucket folder name


    #creating folder locally
    if not os.path.exists(f'{local_path}/{timestamp}'):
        os.makedirs(f'{local_path}/{timestamp}')

    #downloading data to local folder
    command = "gsutil -m cp -r gs://{bucketname}/{blobpath} {localpath}/{timestamp}".format(bucketname = bucket_name,
                                                                          blobpath = blob_path,
                                                                          localpath = local_path,
                                                                          timestamp=timestamp)
    os.system(command)

    return command


def wav_to_raw1D(wave_path, wanted_length= 90000, save=False,sr = None, timestamp = time.strftime('%d_%H_%M') ):

    '''
    Converting a wav file to ndarray and optional saving.
    """
    '''
    # wav to np.array
    sig, srate = librosa.load(wave_path, sr = sr)

    if save:
        new_path = f"{'/'.join(wave_path.split('.')[0].split('/')[:-1])}/raw_1D/{wave_path.split('.')[0].split('/')[-1]}.npy"
        np.save(new_path, sig)
        blob_path = wave_path.split('/')[-1].split('.')[0] #define name of processed wave file eg: '2530'

        storage_client = storage.Client(project=PROJECT)
        bucket = storage_client.get_bucket(BUCKET_NAME)

        #locate file in dedicated folder named after timestamp
        blob = bucket.blob(f"raw_1D/{timestamp}/{blob_path}.npy")
        #blob.upload_from_filename(new_path)
        #os.remove(wave_path)



    return sig
