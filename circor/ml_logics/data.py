import numpy as np
import librosa
import librosa.display
import pandas as pd
import glob
import os

#local
csv_path=os.getcwd()+'/processed_data/df_new.csv'
df_new=pd.read_csv(csv_path)

#from very beginning
#df=pd.read_csv(download_to_local()[1])

def rgba_data(save=True):
    """turning .wave data to rgba of size(224, 224, 3)"""

    X_raw=[]
    for wave_path in glob.glob(f'/Users/fabianlaw/code/fablaw/circor/processed_data/wav_files/*.wav'):
        x, sr=librosa.load(wave_path)

        D = librosa.stft(x[0:50000], n_fft=446, hop_length=224)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        spectrogram = librosa.display.specshow(S_db, y_axis="log", sr=sr, hop_length=1024, x_axis="time")

        rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(S_db.shape))

        rgba=rgbas[:,:,0:3]

        if save:
            new_path = f"{'/'.join(wave_path.split('.')[0].split('/')[:-1])}/X_raw/{wave_path.split('.')[0].split('/')[-1]}.npy"
            np.save(new_path, rgbas)

        X_raw.append(rgba)

    X=np.stack(X_raw)

    y=df_new.outcome.map({'Abnormal': 1, 'Normal': 0})

    return X, y

def rgba_new():

    patient_recordings_list = (df_new['patient_id'].astype(str) +'_'+df_new['audible']).tolist()

    patient_new=np.random.choice(patient_recordings_list)
    print(patient_new)

    file=f'/Users/fabianlaw/code/fablaw/circor/processed_data/wav_files/{patient_new}.wav'
    x, sr=librosa.load(file)

    D = librosa.stft(x[0:50000], n_fft=446, hop_length=224)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    spectrogram = librosa.display.specshow(S_db, y_axis="log", sr=sr, hop_length=1024, x_axis="time")

    rgbas= spectrogram.to_rgba(spectrogram.get_array().reshape(S_db.shape))

    rgba=rgbas[:,:,0:3]

    X_new=rgba.reshape((1,224,224,3))

    return X_new
