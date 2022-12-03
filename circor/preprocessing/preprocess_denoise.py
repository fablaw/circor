import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
#from scipy.io import wavfile
import glob
#import wfdb
import time
#from scipy import signal --> Conventional filters: Kalman filter, Chebyshev filter(even degree cosine polynomial basis as in Chebyshev polynomials?)
#, Butterworth filter; separate notebook
#import noisereduce as nr --> separate notebook
#import tsmoothie --> separate notebook
import math
import pywt #Installation: pip install PyWavelets --> For wavelet based reconstructions
from sympy import Symbol, solve, nsolve, log, N, evalf #Installation: pip install sympy
# --> For higher precision lmbda since librosa.load gives 9 decimal digits (float32)
import soundfile as sf




def select_patients(df: pd.DataFrame, drop_dup: bool = True, all_locations: bool = True, murmur:bool = True) -> pd.DataFrame:
    """
    ABCD
    """
   # circor = pd.read_csv(csv_file)
    circor = df
    circor = circor.drop(columns = ['Pregnancy status', 'Age', 'Sex', 'Height', 'Weight',
       'Pregnancy status',  'Systolic murmur timing',
       'Systolic murmur shape', 'Systolic murmur grading',
       'Systolic murmur pitch', 'Systolic murmur quality',
       'Diastolic murmur timing', 'Diastolic murmur shape',
       'Diastolic murmur grading', 'Diastolic murmur pitch',
       'Diastolic murmur quality', 'Campaign'])
    if drop_dup:
        circor = drop_duplicates(circor)                #Replace with Fabian's random selection of one row from the repetition
        circor = circor.drop(columns='Additional ID')

    if all_locations:
        circor=circor[circor['Recording locations:']=='AV+PV+TV+MV']
    else:
        patient_series = list()
        location_series  = list()
        outcome_series = list()
        for index in range(circor.shape[0]):
            split_locations = circor.iloc[index]['Recording locations:'].split('+')
            location = np.random.choice(split_locations)
            patient_series.append(circor.iloc[index]['Patient ID'])
            location_series.append(location)
            outcome_series.append(circor.iloc[index]['Outcome'])

    if murmur:
        circor=circor[~(circor['Murmur']=='Unknown')]

    circor['Outcome'] = circor['Outcome'].map({'Normal': 0, 'Abnormal':1 })

    tmp_dict = {'Patient ID': patient_series, 'Recording_location': location_series, 'Outcome':outcome_series}
    circor_sample = pd.DataFrame(data=tmp_dict, index=None)
    return circor


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """ABCD"""
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


def synchronise(npy_array: np.ndarray, tsv_df: pd.DataFrame, sr: int = 4000, num_cycles: int = None) -> np.ndarray:
    """
    Slice the numpy arrays keeping only the non-zero time segments (i.e. 1, 2, 3, 4) from the corresponding tsv file;
    Edit: Shall keep the missing segments (labelled 0 parts) and the corresponding non-zero segments in between
    Save the corresponding audio file, np.array to cloud? Cloud only for the preprocessed output for now
    """
    for j in tsv_df.index:
        if tsv_df.iloc[j][2] == 1:
            row_start = j
            break
    if not num_cycles:
        for k in tsv_df.index:
            if tsv_df.iloc[tsv_df.shape[0] - k][2] == 4:
                row_end = tsv_df.shape[0] - k
    else:
        row_end = row_start + num_cycles * 4  #Erroneous due to the partial segmentation labelling, but acceptable


    tsv_df = tsv_df[row_start:row_end]          #Certain files do not have 1 as the second segment;
                                                #would need to use an if condition; probably already in Fabian's
    start_index = math.ceil(sr * tsv_df.iloc[0][0])
    end_index = math.floor(sr * tsv_df.iloc[-1][1]) + 1
    npy_array_sync = npy_array[start_index:end_index]  #May need to correct the values by +/- 1, but only minor effects

    return npy_array_sync



def reconstruct_signal(sig: pd.Series, wavelet: str = 'db11', level: int = None, mode: str = 'antireflect', sigma: float =0.02,\
    sig_len = None, sig_start = 0) -> pd.Series :
    """ABCD"""

    #Apply identical reconstructions to all signals, later try to find one adapted to each signal using a for loop and a noise measuring fn

    if sig_len:
            sig = sig[sig_start: sig_start + sig_len]  #try except for the "Array out of bounds" case
    else:
        sig = sig[sig_start:]

    coeffs = pywt.wavedec(data=sig,wavelet= wavelet, level = level, mode=mode)
    #Daubechies (daub) 11, 14,and 20; Symlet (sym) 9, 11, and 14; Coiflet (???) 4 and 5; subsets

    #Default level = max, usually 10 or 11; 5 seems sufficient; 8 seems ok
    #Default mode = 'symmetric'; 'antireflect' and 'periodize' seem appropriate
    approx = coeffs[0]
    details = coeffs[1:]

    details_nb = neigh_block(details, sig.shape[0] , sigma = sigma) #sigma is a noise level;
                                                                    #unclear how to choose; Gaussian white noise assumed;
                                                                    #somewhat inaccurate assumption for our system
    sig_dn = pywt.waverec([approx] + details_nb, wavelet= wavelet)
    return sig_dn



def neigh_block(details, n, sigma):
    """ABCD"""
    res = list()
    L0 = math.floor(np.log2(n) / 2)
    L1 = max(1, math.floor(L0 / 2))
    L = L0 + 2 * L1

    for d in details:
        d2 = d.copy()
        for start_b in range(0, len(d2), L0):
            end_b = min(len(d2), start_b + L0)
            start_B = start_b - L1
            end_B = start_B + L
            if start_B < 0:
                end_B -= start_B
                start_B = 0
            elif end_B > len(d2):
                start_B -= end_B - len(d2)
                end_B = len(d2)
            assert end_B - start_B == L
            d2[start_b:end_b] *= nb_beta(d2[start_B:end_B], L, sigma)
        res.append(d2)
    return res


def nb_beta(detail, L, sigma):
    """!!!!ABCD!!!!"""
    z = Symbol('z')
    sol_set = solve(z - log(z) - 3, z, dict=True)
    accuracy = 10
    lmbd = float(N(sol_set[1][z], accuracy)) #May simply replace by the value directly
    S2 = np.sum(detail ** 2)
    beta = (1 - lmbd * L * sigma**2 / S2)
    return max(0, beta)


# def save_treated_npy_wav_files(wavelet = 'db11', level = None, mode = 'antireflect', sigma=0.02):
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



for file in glob.glob('../raw_data/training_data/*.npy'):
    npy_file = np.load(file)
    tsv_file = file.replace('npy','tsv')
    tsv_file = pd.read_csv(tsv_file, sep = '\t',header=None)
    npy_file_sync = synchronise(npy_file,tsv_file)
    np.save(file.replace('training_data','npy_synchronised_unpadded'), npy_file_sync)



def tmp_fn(df, index, array_length, all_locations = True):
    """ABCD FOR THE ONE FEATURE CASE"""
    circor = df
    patient_id  = circor.iloc[index]["Patient ID"]
    rec_loc = circor.iloc[index]["Recording_location"]
 #   for file_path in glob.glob(os.path.join(path, f'{patient_id}_{rec_loc}'+'*.npy')):
        #Only one element, so can also use glob.glob(os.path.join(path, f'{patient_id}_{rec_loc}'+'*.npy'))[0]

    file_path = glob.glob(os.path.join(npy_path, f'{patient_id}_{rec_loc}'+'*.npy'))[0]
    tmp_array = np.load(file_path)
    tmp_array = tmp_array[:array_length]


    return tmp_array



def processed_df(df, time_series = True, array_length=6_000):
    """ABCD"""
    circor = df
    circor['numpy_arrays'] =[tmp_fn(circor, index, array_length=array_length) for index in circor.index]
    #Without defining it as an np.array explicitly



 #   circor['AV'] =

    #X = circor.loc[:, 'AV', 'PV', 'TV', 'MV']



    X = circor.loc[:, 'numpy_arrays']
    X = np.stack(X)
    X = np.expand_dims(X, axis=2)
    y = circor[['Outcome']]




    if time_series:
        mean_mean = np.mean(np.mean(np.abs(X_train), axis = 0))
        std_mean = np.mean(np.std(np.abs(X_train), axis = 0, ddof=0))
        scaling_factor = 1 / (mean_mean + std_mean)
        X *= scaling_factor
   # padding with mask value = -10!!!
    return X, y
