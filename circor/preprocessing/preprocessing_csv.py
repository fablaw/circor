import numpy as np
import pandas as pd
import os
from circor.preprocessing.main_1D import download_to_local

project=os.environ.get("PROJECT")
bucket_name=os.environ.get("BUCKET_NAME")

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
    df_new=pd.DataFrame({'patient_id': df['Patient ID'],
                         'select': random_rec,
                         'audible': df['Most audible location'],
                         'outcome': df.Outcome
                        })

    df_new.audible.fillna(df_new.select, inplace=True)

    return df_new

def select_patients():

    """
    Select rows based on returning patients, all or partial locations measured, murmur status. It returns a modified features df and the
    outcome series
    """
    #retrieving data from google cloud

    #reading data saved locally
    df=pd.read_csv(download_to_local())

    #remove rows with unknown murmur
    df_1=df[~df['Murmur'].isin(['Unknown'])]

    #remove duplicates(appear in both campaigns)
    df_2=drop_duplicates(df_1)

    #selecting only subjects with all recordings at 4 different locations
    df_3=df_2[df_2['Recording locations:']=='AV+PV+TV+MV']

    #select most audible location if available, otherwise, assign random choice
    df_4=select_1_recording(df_3)

    #save to local
    if not os.path.exists(f'circor/processed_data'):
        os.makedirs(f'circor/processed_data')

    filepath=f'circor/processed_data/df_new.csv'
    df_4.to_csv(filepath, index=False, header=True)

    print("\n✅ CSV preprocessed!")

    return df_4

if __name__ == '__main__':
    drop_duplicates()
    select_1_recording()
    select_patients()
