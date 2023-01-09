import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import glob
import time
import math
import pywt #Installation: pip install PyWavelets --> For wavelet based reconstructions
from sympy import Symbol, solve, nsolve, log, N, evalf #Installation: pip install sympy
# --> For higher precision lmbda since librosa.load gives 9 decimal digits (float32); May drop later
#import soundfile as sf --> To save the processed the series; for testing and may be demonstrations
from google.cloud import storage
from circor.parameters.params import BUCKET_NAME, PROJECT
#from tensorflow.keras.preprocessing.sequence import pad_sequences


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop all patients who have an Additional ID"""
    doublon=data[['Patient ID','Additional ID']].dropna()
    liste_couple=[]
    for i in range(len(doublon)):
        min_id=min(doublon.iloc[i]['Patient ID'],doublon.iloc[i]['Additional ID'])
        max_id=max(doublon.iloc[i]['Patient ID'],doublon.iloc[i]['Additional ID'])
        if [min_id,max_id] not in liste_couple:
            liste_couple.append([min_id,max_id])
    list_id_drop=np.array(liste_couple)[:,1]
    data_drop_dup=data[~data['Patient ID'].isin(list_id_drop)]
    return data_drop_dup


# def save_treated_npy_wav_files(wavelet: str = 'db11', level: int = None, mode: str = 'antireflect', sigma: float=0.02) -> np.ndarray:
#     """ABCD"""
#     for file_path in glob.glob(os.path.join(npy_path,f'*.npy')):
#         for patient_id in circor_dropped['Patient ID']:
#             if str(patient_id) in file_path:
#                 sig = np.load(file_path)
#                 sig_dn = reconstruct_signal(sig, wavelet = wavelet, level = level, mode = mode, sigma=sigma)
#                 tmp_file_path_npy = file_path.replace('npy_synchronised_unpadded', 'npy_synchronised_unpadded_treated')
#                 tmp_file_path_wav = file_path.replace('npy_synchronised_unpadded', 'audio_treated').replace('npy','wav')
#                 np.save(tmp_file_path_npy, sig_dn)
#                 sf.write(file=tmp_file_path_wav, data=sig_dn, samplerate=4000, subtype='PCM_24')
#     return sig_dn



# for file in glob.glob('../raw_data/training_data/*.npy'):
#     npy_file = np.load(file)
#     tsv_file = file.replace('npy','tsv')
#     tsv_file = pd.read_csv(tsv_file, sep = '\t',header=None)
#     npy_file_sync = synchronise(npy_file,tsv_file)
#     np.save(file.replace('training_data','npy_synchronised_unpadded'), npy_file_sync)


def preprocess_sig(df, drop_dup = True, all_locations = False, array_length=6_000, time_series:bool = True):
    """ABCD"""
    X = select_patients(df,drop_dup=drop_dup, all_locations=all_locations, murmur=True)[0]
    #Only for the restricted cases now; single location case
    y = select_patients(df,drop_dup=drop_dup, all_locations=all_locations, murmur=True)[1]

    X['numpy_arrays'] =[download_reconstruct_upload(X, index, array_length=array_length) for index in X.index]
    #Without defining it as an np.array explicitly



    #  circor['AV'] =

    #X = circor.loc[:, 'AV', 'PV', 'TV', 'MV']



    X = X.loc[:, 'numpy_arrays']
    X = np.stack(X)
    X = np.expand_dims(X, axis=2)

    #For the direct time series case
    #X = pad_sequences(X, dtype='float32', padding='post', value=-10)

    # padding with mask value = -10

    #For the train_test_split step
    # mean_mean = np.mean(np.mean(np.abs(X_train), axis = 0))
    # std_mean = np.mean(np.std(np.abs(X_train), axis = 0, ddof=1)) #Sample std
    # scaling_factor = 1 / (6*(mean_mean + std_mean))  # so that we get a factor of around 3; the mean and std are not
    # that changed by the noise treatment; consistent with the (white) Gaussian noise with low std assumption (?)
    # X *= scaling_factor

    return X, y
