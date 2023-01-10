import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os


#use the full page
st.set_page_config(layout="wide")

st.sidebar.title('Welcome to CardioBot!🩺 \n  ')
st.sidebar.subheader('Your autonomous heartbeat checkup')

st.sidebar.write('Dear Patient,')
st.sidebar.write("On this online platform \
                you will be able to run preliminary health checks \
                before going to your cardiologist.")
st.sidebar.markdown('---')
st.sidebar.title('How does it work ?')
st.sidebar.write(" 1) Cardio bot will first convert your heart recording into a spectrogram. \
                This way, you will be able to see a visual representation of your heartbeat.")
st.sidebar.write(" 2) In a second step, Cardio bot will \
                detect if your heart has a murmur.")
phonocardiogram_raw = st.sidebar.file_uploader(label="Upload your phonocardiogram below :",
                                           type='.wav')

#allow nested buttons
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage


submit_button = st.sidebar.button(label='How is my heart looking 💓? ', on_click=set_stage, args=(1,))

if st.session_state.stage > 0:

    st.title(body='CardioBot app')

    st.subheader('Visualize my heart checkup')

    #display 3 visualisations of raw data prior to denoising

    #Make visuals 3x bigger than audio
    col1, col2, col3 = st.columns([1,3,3])

    #Phonocardiogram of raw data
    col1.markdown("Original Audio file of your heartbeat")
    col1.audio(phonocardiogram_raw, format="audio.wav", start_time=0)

    #Spectrogram of raw data
    col2.markdown("Spectrogram of your heartbeat")
    phonocardiogram, fs = librosa.load(phonocardiogram_raw, sr=None)
    phonocardiogram = np.array(phonocardiogram)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(phonocardiogram)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, sr=fs,x_axis='time', y_axis='log')
    plt.xlabel('Time in seconds')
    plt.xlim(left= 0,right= 2)
    col2.pyplot(fig)

    #Oscillogram of raw data
    col3.markdown("phonocardiogram of your heartbeat")
    fig_2, ax = plt.subplots()
    img_oscillo = plt.plot(phonocardiogram)
    plt.xlim(left= 0,right= 5000)
    plt.ylim(bottom=-0.5, top=0.5)
    col3.pyplot(fig_2)

    st.button('Clean my Phonocardiogram please!', on_click=set_stage, args=(2,))
    st.markdown("___")

    if st.session_state.stage > 1:

        st.subheader('Visualize my heart checkup with less noise')

        col1, col2, col3 = st.columns([1,3,3])

        phonocardiogram_clean = open(f'{os.getcwd()}/circor/processed_data/wav_files/{phonocardiogram_raw.name}', 'rb')
        audio_bytes = phonocardiogram_clean.read()
        col1.audio(audio_bytes, start_time=0)

        #Spectrogram of clean data
        col2.markdown("Spectrogram of your heartbeat")
        phonocardiogram_npy_clean = np.load(f'{os.getcwd()}/circor/processed_data/npy_files/{phonocardiogram_raw.name[:-4]}.npy')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(phonocardiogram_npy_clean)), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D, sr=fs,x_axis='time', y_axis='log')
        plt.xlabel('Time in seconds')
        plt.xlim(left= 0,right= 1.5)
        col2.pyplot(fig)

        #Oscillogram of raw data
        col3.markdown("phonocardiogram of your heartbeat")
        fig_2, ax = plt.subplots()
        img_oscillo = plt.plot(phonocardiogram_npy_clean)
        plt.xlim(left= 0,right= 5000)
        plt.ylim(bottom=-0.5, top=0.5)
        col3.pyplot(fig_2)




    st.button('Should I go see the cardiologist ?', on_click=set_stage, args=(3,))
    st.markdown("___")
    if st.session_state.stage > 2:

        prediction_dict = {'13918_TV': '50 %',
                           '14241_PV': '55 %',
                           '85308_AV': '58 %',
                           '85337_MV': '51 %',
                           '85343_TV': '62 %'}

        st.markdown("<h1 style='text-align: center; color: blue; font-size:300%;'>Yes, please</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: left; font-size:150%;'>Dear patient,</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns((3,1,3))
        col1.write("<h1 style='text-align: left; font-size:150%;'>Having analysed your heartbeat recording, \
                 we found that there are  chances that you have a murmur.</h1>", unsafe_allow_html=True)
        col1.write("<h1 style='text-align: left; font-size:150%;'>We would like to recommend you to go see your cardiologist!</h1>", unsafe_allow_html=True)
        col2.write('')
        col3.write("<h1 style=' font-size:150%;'>Probability of having a murmur :</h1>", unsafe_allow_html=True)
        col3.subheader(f"{prediction_dict[phonocardiogram_raw.name[:-4]]}")
        st.write("<h1 style='text-align: center; color: blue; font-size:200%;'>See you soon on CardioBot, take care!</h1>", unsafe_allow_html=True)
