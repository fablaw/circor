import numpy as np
def gray_padd(file,max_length):
    """take a 2D_array rgbas file as input and return a gray and padded to max_length file"""
    file_gray = np.mean(file,axis=2).astype('uint8')
    if file_gray.shape[1]>max_length:
        pad = file_gray[:,:max_length]
    else:
        pad = np.pad(file_gray,pad_width=((0,0),(0, max_length - file_gray.shape[1])), mode = 'constant', constant_values = -10)
    return pad

def create_X(data_csv,max_length):
    X_pre=np.empty([len(data_csv),1025, max_length,4])
    iteration=0
    for i in data_csv['Patient ID']:
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
    return X_pre
