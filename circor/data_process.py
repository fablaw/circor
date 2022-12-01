import numpy as np
import pandas as pd

def drop_duplicates(data_csv):
    """drop the second occurence of a patient while keeping the first one"""

    doublon=data_csv[['Patient ID','Additional ID']].dropna()
    liste_couple=[]
    for i in range(len(doublon)):
        min_id=min(doublon.iloc[i]['Patient ID'],doublon.iloc[i]['Additional ID'])
        max_id=max(doublon.iloc[i]['Patient ID'],doublon.iloc[i]['Additional ID'])
        if [min_id,max_id] not in liste_couple:
            liste_couple.append([min_id,max_id])
    list_id_drop=np.array(liste_couple)[:,1]
    data_csv_drop_dup=data_csv[~data_csv['Patient ID'].isin(list_id_drop)]
    return data_csv_drop_dup

def clean_data_csv(data_csv,choice_locations='all'):
    """from data_csv get clean data: encode targets,drop duplicate
    and choose patient where we have the locations wanted"""
    df_clean=drop_duplicates(data_csv)
    if choice_locations=='all':
        df_clean=df_clean[df_clean['Recording locations:']=="AT+PV+TV+MV"]
    if choice_locations=='one':
        ls=[]
        for ind in df_clean.index:
            r=df_clean.loc[ind, 'Recording locations:'].split('+')
            l=np.random.choice(r)
            ls.append(l)
        df_new=pd.DataFrame({'Patient_id': df_clean['Patient ID'],
                        'select': ls,
                        'audible': df_clean['Most audible location']
                        })
        df_new.audible.fillna(df_new.select, inplace=True)
        df_clean['Recording locations']=df_new['audible']
    if choice_locations=='only audible murmur':
        df_clean=df_clean[df_clean['Most audible location'!=np.nan]]
        df_clean['Recording locations']=df_new['Most audible locations']
    df_clean['Outcome'].replace(to_replace='Abnormal',value=1,inplace=True)
    df_clean['Outcome'].replace(to_replace='Normal',value=0,inplace=True)
    return df_clean
