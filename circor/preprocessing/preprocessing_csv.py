import numpy as np
import pandas as pd
import os
# import librosa
# import librosa.display
import glob
import time
import math
# import pywt #Installation: pip install PyWavelets --> For wavelet based reconstructions
# from sympy import Symbol, solve, nsolve, log, N, evalf #Installation: pip install sympy
# # --> For higher precision lmbda since librosa.load gives 9 decimal digits (float32); May drop later
# #import soundfile as sf --> To save the processed the series; for testing and may be demonstrations
from google.cloud import storage
from circor.parameters.params import BUCKET_NAME, PROJECT
#from tensorflow.keras.preprocessing.sequence import pad_sequences


def select_patients(df: pd.DataFrame, drop_dup: bool = True, all_locations: bool = True, murmur:bool = True) ->pd.DataFrame:
#-> tuple[pd.DataFrame, pd.Series]:
    """
    Select rows based on returning patients, all or partial locations measured, murmur status. It returns a modified features df and the
    outcome series
    """

    circor = df
    circor = circor.drop(columns = ['Pregnancy status', 'Age', 'Sex', 'Height', 'Weight',
       'Pregnancy status',  'Systolic murmur timing',
       'Systolic murmur shape', 'Systolic murmur grading',
       'Systolic murmur pitch', 'Systolic murmur quality',
       'Diastolic murmur timing', 'Diastolic murmur shape',
       'Diastolic murmur grading', 'Diastolic murmur pitch',
       'Diastolic murmur quality', 'Campaign'])
    circor['Outcome'] = circor['Outcome'].map({'Normal': 0, 'Abnormal':1 })

    if drop_dup:
        circor = drop_duplicates(circor)                #Replace with Fabian's random selection of one row from the repetition
        circor = circor.drop(columns='Additional ID')

    if murmur:
        circor=circor[~(circor['Murmur']=='Unknown')]

    if all_locations:
        circor=circor[circor['Recording locations:']=='AV+PV+TV+MV']
        X = circor[['Patient ID', 'Recording locations']]
    else:
        # patient_series = list()
        # location_series  = list()
        # outcome_series = list()
        # for index in range(circor.shape[0]):
        #     split_locations = circor.iloc[index]['Recording locations:'].split('+')
        #     location = np.random.choice(split_locations, size =1, p= None ) #Not yet picking the most audible location; or weights
        #     patient_series.append(circor.iloc[index]['Patient ID'])
        #     location_series.append(location)
        #     outcome_series.append(circor.iloc[index]['Outcome'])
        # tmp_dict = {'Patient ID': patient_series, 'Recording_location': location_series, 'Outcome':outcome_series}
        # circor = pd.DataFrame(data=tmp_dict, index=None)
        circor = select_1_recording(circor)
     #   X = circor[['Patient ID', 'Recording_location']]
    #y = circor['Outcome']

  # return X, y
    return circor

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

def select_1_recording(df: pd.DataFrame) -> pd.DataFrame:
    """select one recording: either the one where the murmur is most audible or random one"""
    random_rec=[]
    for ind in df.index:
        rec=df.loc[ind, 'Recording locations:'].split('+')
        location=np.random.choice(rec)
        random_rec.append(location)

    #construct new dataframe out of cleaned dataframe
    #FIX INDENDATION ERROR!!!!
    df_new=pd.DataFrame({'Patient ID': df['Patient ID'],
                        'select': random_rec,
                        'Most audible location': df['Most audible location'],
                        'Outcome': df.Outcome
                        })
    df_new['Most audible location'].fillna(df_new.select, inplace=True)
    df_new.rename(columns = {'audible: Recording_location'})
    return df_new
