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
import soundfile as sf #Pip install soundfile ??




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
            if tsv_df.iloc[tsv_df.shape[0] - k -1][2] == 4:
                row_end = tsv_df.shape[0] - k
                break
    else:
        row_end = row_start + num_cycles * 4  #Erroneous due to the partial segmentation labelling, but acceptable


    tsv_df = tsv_df[row_start:row_end]          #Certain files do not have 1 as the second segment;
                                                #would need to use an if condition; probably already in Fabian's
    start_index = math.ceil(sr * tsv_df.iloc[0][0])
    end_index = math.floor(sr * tsv_df.iloc[-1][1]) + 1
    npy_array_sync = npy_array[start_index:end_index]  #May need to correct the values by +/- 1, but only minor effects

    return npy_array_sync




def reconstruct_signal(sig: np.ndarray, tsv_df: pd.DataFrame, wavelet: str = 'db11', level: int = None, mode: str = 'antireflect', sigma: float =0.02,
                       sig_len = None, sig_start = 0,  sr: int = 4000, num_cycles: int = None) -> pd.Series :
    """
    Call the synchronise function to obtain a time series starting with the "1" segment.
    Then apply the wave reconstruction to return a noise treated series
    """

    #Apply identical reconstructions to all signals, later try to find one adapted to each signal using a for loop and a noise measuring fn
    #Using ragged arrays seem to make them no longer ndarrays but general objects; not safe to vectorize the transformations;
    #Padding only at the end, as putting masking conditions here seem indirect

    sig = synchronise(npy_array=sig,tsv_df=tsv_df, sr=sr, num_cycles=num_cycles)

    if sig_len:
            sig = sig[:, sig_start: sig_start + sig_len]  #try except for the "Array out of bounds" case
    else:
        sig = sig[:, sig_start:]

    coeffs = pywt.wavedec(data=sig, wavelet= wavelet, level = level, mode=mode)
    #Daubechies (daub) 11, 14,and 20; Symlet (sym) 9, 11, and 14; Coiflet (???) 4 and 5; subsets

    #Default level = max, usually 10 or 11; 5 seems sufficient; 8 seems ok
    #Default mode = 'symmetric'; 'antireflect' and 'periodize' seem appropriate
    approx = coeffs[0]
    details = coeffs[1:]

    details_nb = neigh_block(details, sig.shape[0] , sigma = sigma) #sigma is a noise level;
                                                                    #unclear how to choose; Gaussian white noise assumed;
                                                                    #somewhat inaccurate assumption for our system
    coeffs = np.concatenate(np.array([approx]), details_nb)
    sig_dn = pywt.waverec(coeffs=coeffs, wavelet= wavelet, mode=mode, axis=-1)
    return sig_dn



def neigh_block(details: np.ndarray, n: int, sigma: float) -> np.ndarray:
    """Calculate a new array of details based on the Neighbouring Blocks method:
    Incorporating Information on Neighbouring Coefficients into Wavelet Estimation
    T. Tony Cai and Bernard W. Silverman
    https://www.jstor.org/stable/25053168
    to take care of edge effects
    Two blocks of code slightly adapted from:
    https://github.com/CSchoel/learn-wavelets/blob/main/wavelet-denoising.ipynb
    """
    res = list()
    L0 = math.floor(np.log2(n) / 2)
    L1 = max(1, math.floor(L0 / 2))
    L = L0 + 2 * L1
    z = Symbol('z')
    sol_set = solve(z - log(z) - 3, z, dict=True)
    accuracy = 10
    lmbd = float(N(sol_set[1][z], accuracy)) #May simply replace by the value directly

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
            d2[start_b:end_b] *= nb_beta(d2[start_B:end_B], L, sigma, lmbd=lmbd)
        res.append(d2)
    return res


def nb_beta(detail: np.ndarray, L:int, sigma: float, lmbd: float) -> float:
    """
    Find the beta factor for the NeighBlock method
    Taken from:
    https://github.com/CSchoel/learn-wavelets/blob/main/wavelet-denoising.ipynb
    """

    S2 = np.sum(detail ** 2)
    beta = (1 - lmbd * L * sigma**2 / S2)
    return max(0, beta)





def download_reconstruct_upload(df: pd.DataFrame, index: int, array_length: int, all_locations: bool = True,
                                wavelet: str='db14', level:int=None, mode:str='antireflect', sigma:float=0.02,
                                sig_len: int = 6_000, sig_start:int=0,
                                sr:int = 4000, num_cycles: int=5) -> np.ndarray:
    """Download the raw npy file from the cloud and returns it with the selected array length """
    X = df
    patient_id  = X.iloc[index]["Patient ID"]
    rec_loc = X.iloc[index]["Recording_location"]

    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.get_bucket(BUCKET_NAME)

    timestamp = time.strftime('%d_%H_%M')

    if not os.path.exists(f'../raw_data/training_data_gcloud/{timestamp}'):
        os.makedirs(f'../raw_data/training_data_gcloud/{timestamp}')

    blob_npy = bucket.blob(f"../training_data/{patient_id}_{rec_loc}.npy")
    blob_tsv = bucket.blob(f"../tsv_raw/{patient_id}_{rec_loc}.tsv")

    file_path_npy = os.path.join('../raw_data/training_data_gcloud/',f"{patient_id}_{rec_loc}.npy")
    file_path_tsv = os.path.join('../raw_data/training_data_gcloud/',f"{patient_id}_{rec_loc}.tsv")

    blob_npy.download_to_filename(file_path_npy)
    blob_npy.download_to_filename(file_path_tsv)



    sig = np.load(file_path_npy)
    tsv_df = pd.read_csv(file_path_tsv, sep='\t', header = None)

    #Treat and reconstruct the signal
    sig = reconstruct_signal(sig=sig, tsv_df=tsv_df, wavelet=wavelet, level=level, mode=mode, sigma=sigma, sig_len = sig_len, sig_start=sig_start,
                             sr = sr, num_cycles=num_cycles)

    if not os.path.exists(f'../processed_data/npy_files_{timestamp}'):
        os.makedirs(f'../processed_data/npy_files_{timestamp}')
    if not os.path.exists(f'../processed_data/wav_files_{timestamp}'):
        os.makedirs(f'../processed_data/wav_files_{timestamp}')


    #Defining local file paths
    file_path_npy_processed = os.path.join('../processed_data/npy_files_{timestamp}/',f"{patient_id}_{rec_loc}.npy")
    file_path_wav_processed = os.path.join('../processed_data/wav_files_{timestamp}/',f"{patient_id}_{rec_loc}.wav")

    #Save files locally
    np.save(file_path_npy_processed)
    sf.write(file=file_path_wav_processed, data=sig, samplerate=4000, subtype='PCM_24')

    blob_path_npy =  file_path_npy_processed.split('/')[-1].split('.')[0] #define name of processed npy file
    blob_path_wav =  file_path_wav_processed.split('/')[-1].split('.')[0] #define name of processed wav file

    #locate file in dedicated folder named after timestamp
    blob_npy_processed = bucket.blob(f"../processed_data/npy_files/{timestamp}/{patient_id}_{rec_loc}.npy")
    blob_wav_processed = bucket.blob(f"../processed_data/wav_files/{timestamp}/{patient_id}_{rec_loc}.wav")

    #Upload to bucket
    blob_npy_processed.upload_from_filename(file_path_npy_processed)
    blob_wav_processed.upload_from_filename(file_path_wav_processed)


    #Google bucket info
    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.get_bucket(BUCKET_NAME)

    #May need to fix missing arguments in functions
    os.remove(file_path_npy)
    os.remove(file_path_tsv)
    return sig
