
import time
from circor.parameters.params import BUCKET_NAME
from circor.preprocessing.wav_to_1D import download_bucket_objects, wav_to_raw1D
import glob
from circor.preprocessing.preprocessing_csv import select_patients

def wav_to_1D():

    ''' preprocess your recordings to tranform them into 1D arrays -for RNN purposes.'''

    timestamp = time.strftime('%d_%H_%M')



    local_path=download_bucket_objects(bucket_name= BUCKET_NAME,
                            blob_path= 'audio_raw',
                            local_path = f'circor/raw_data',
                            timestamp=timestamp)


    for wave_path in glob.glob(f'circor/raw_data/{timestamp}/audio_raw/*.wav'):
        output = wav_to_raw1D(wave_path,
                                  save=True,
                                  timestamp=timestamp)

    return local_path

def download_to_local():
    #local path for tsv file
    local=download_bucket_objects(bucket_name=BUCKET_NAME,
                            blob_path= 'tsv_raw',
                            local_path = f'circor/raw_data',
                            timestamp=None)

    tsv_filepath=local + '/06_11_04/tsv_raw'

    #local path for csv file
    csv_filepath=select_patients()

    #local path for npy file
    input = wav_to_1D()
    npy_filepath=local + '/06_11_04/audio_raw'

    return (tsv_filepath, csv_filepath, npy_filepath)
