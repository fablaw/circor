import numpy as np
import glob
import librosa
import pandas as pd

#2D data process

def gray_padd(file,max_length):
    """take a 2D_array rgbas file as input and return a gray and padded to max_length file"""
    file_gray = np.mean(file,axis=2).astype('uint8')
    if file_gray.shape[1]>max_length:
        pad = file_gray[:,:max_length]
    else:
        pad = np.pad(file_gray,pad_width=((0,0),(0, max_length - file_gray.shape[1])), mode = 'constant', constant_values = -10)
    return pad

def create_X(clean_data_csv,max_length):
    if clean_data_csv['Recording locations'=='AT+PV+TV+MV']:
        X_pre=np.empty([len(clean_data_csv),1025, max_length,4])
        iteration=0
        for i in clean_data_csv['Patient ID']:
            i_all=np.empty([1025, max_length,4])

            file_1='../content/processed_2D/' + str(i) + '_AV.wav.npy'
            av = np.load(file_1)
            av=gray_padd(av,max_length)
            i_all[:,:,0]=av

            file_2='../content/processed_2D/' + str(i) + '_MV.wav.npy'
            mv = np.load(file_2)
            mv=gray_padd(mv,max_length)
            i_all[:,:,1]=mv

            file_3='../content/processed_2D/' + str(i) + '_PV.wav.npy'
            pv = np.load(file_3)
            pv=gray_padd(pv,max_length)
            i_all[:,:,2]=pv

            file_4='../content/processed_2D/' + str(i) + '_TV.wav.npy'
            tv = np.load(file_4)
            tv=gray_padd(tv,max_length)
            i_all[:,:,3]=tv

            X_pre[iteration,:,:,:]=i_all
            iteration+=1
    else:
        X_pre=np.empty([len(clean_data_csv),1025, max_length])
        iteration=0
        for i in clean_data_csv.index:
            path='../content/processed_2D/' + str(clean_data_csv.Patient_id[i]) + '_' + str(clean_data_csv['Recording locations'][i])+'.wav.npy'
            file = np.load(path)
            file=gray_padd(file,max_length)
            X_pre[iteration,:,:]=file
            iteration+=1

    return X_pre


#1D data process

def process_5_cycles_synchronised():
    name, start, duration=[],[],[]
    for f in glob.glob(f'raw_data/training_data/*.tsv'):
        x=(''.join(f.rsplit('.tsv',1))).split('/')
        name.append(x[-1])
        cs=pd.read_csv(f, sep='\t')
        sta=cs.iloc[0,0]
        sto=cs.iloc[19,1]
        dura=sto-sta
        start.append(sta)
        duration.append(dura)

    for index in range(len(name)):
        file='raw_data/training_data/'+name[index]+'.wav'
        x_1, fs=librosa.load(file, sr=None, offset=start[index], duration=duration[index])

        output='/Users/fabianlaw/code/fablaw/circor/raw_data/1.0.3/npy_sync_5_cycles/'+name[index]+'.npy'
        np.save(output,x_1)

def select_1_recording(df_drop_dup):
    """select one recording: either the one where the murmur is most audible or random one"""
    ls=[]
    for ind in df_drop_dup.index:
        r=df_drop_dup.loc[ind, 'Recording locations:'].split('+')
        l=np.random.choice(r)
        ls.append(l)
    df_new=pd.DataFrame({'Patient_id': df_drop_dup['Patient ID'],
                     'select': ls,
                     'audible': df_drop_dup['Most audible location']
                    })
    df_new.audible.fillna(df_new.select, inplace=True)
