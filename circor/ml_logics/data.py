import numpy as np
import librosa
import librosa.display
import pandas as pd
import os
from circor.preprocessing.preprocessing_csv import select_patients
from google.cloud import storage
import glob


project=os.environ.get("PROJECT")
bucket_name=os.environ.get("BUCKET_NAME")
storage_client = storage.Client(project=project)
bucket = storage_client.get_bucket(bucket_name)

def rgba_data(save=False):
    """turning .wave data to rgba of size(224, 224, 3)"""
    df_new=select_patients()

    blob = bucket.blob(f"audio_treated_02_12_2022")

    if not os.path.exists(f'../processed_data'):
        os.makedirs(f'../processed_data')

    local_storage='../processed_data'
    blob.download_from_filename(local_storage)

    X_raw=[]
    for i in df_new.index:
        file='../processed_data/audio_treated_02_12_2022/'+str(df_new.loc[i, 'Patient ID'])+'_'+df_new.loc[i,'Recording_location']+'.wav'
        x, sr=librosa.load(file)

        D = librosa.stft(x[0:50000], n_fft=446, hop_length=224)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        spectrogram = librosa.display.specshow(S_db, y_axis="log", sr=sr, hop_length=1024, x_axis="time")

        rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(S_db.shape))

        rgba=rgbas[:,:,0:3]

        X.raw.append(rgba)

        if save==True:
            output='/processed_data/'+str(df_new.loc[i, 'Patient ID'])+'_'+df_new.loc[i,'Recording_location']+'.npy'
            np.save(output, rgba)

    X=np.stack(X_raw)

    y=df_new.Outcome

    return (X, y)

def rgba_new():

    blob = bucket.blob(f"")

    if not os.path.exists(f'../new'):
        os.makedirs(f'../new')

    local_storage="../new"
    blob.download_from_filename(local_storage)

    file=glob.glob(f'/Users/fabianlaw/code/fablaw/circor/new/*.wav')
    x, sr=librosa.load(file)

    D = librosa.stft(x[0:50000], n_fft=446, hop_length=224)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    spectrogram = librosa.display.specshow(S_db, y_axis="log", sr=sr, hop_length=1024, x_axis="time")

    rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(S_db.shape))

    rgba=rgbas[:,:,0:3]

    return rgba.reshape((1,224,224,3))
