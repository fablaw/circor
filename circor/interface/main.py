from circor.preprocessing.preprocess_cnn import wav_to_2D_HD, wav_to_2D_compressed
from circor.preprocessing.preprocess_rnn import wav_to_1D_padded, get_max_length
import glob


def wav_to_2D(compressed_data=True):

    ''' preprocess your recordings to tranform them into 2D arrays -for CNN purposes
        2 options available : saving recordings in high quality or under uint8 format (latter is quicker)  '''
    if compressed_data:

        for wave_path in glob.glob('raw_data/training_data/*.wav'):
            output = wav_to_2D_compressed(wave_path, hop_length=1024, save=True)
    else:

        for wave_path in glob.glob('raw_data/training_data/*.wav'):
            output = wav_to_2D_HD(wave_path, hop_length=1024, save=True)

    return

def wav_to_1D():

    ''' preprocess your recordings to tranform them into 1D arrays -for RNN purposes.
        padding is done based on the maximum length available in the training set'''

    max_length = get_max_length()

    for wave_path in glob.glob('raw_data/training_data/*.wav'):
        output = wav_to_1D_padded(wave_path, save=True, wanted_length=max_length)

    return





if __name__ == '__main__':

    wav_to_2D(compressed_data=True)
    wav_to_1D()
