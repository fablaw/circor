import streamlit as st
import requests
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

#use the full page
st.set_page_config(layout="wide")

st.sidebar.title('Welcome on Cardio bot!🩺 \n  ')
st.sidebar.subheader('Your autonomous heartbeat checkup')

st.sidebar.write('Dear Patient,')
st.sidebar.write("On this online platform \
                you will be able to run preliminary health checks \
                before going to your cardiologist.")
st.sidebar.markdown('---')
st.sidebar.title('How is it working ?')
st.sidebar.write(" 1) Cardio bot will first convert your heart recording into a spectogram. \
                This way, you will be able to see a visual representation of your heartbeat")
st.sidebar.write(" 2) In a second step, Cardio bot will \
                detect if your heart has a murmur or not")
phonocardiogram = st.sidebar.file_uploader(label="Upload your phonocardiogram below :",
                                           type='.wav')

with st.sidebar.form(key='How is my heart looking ?'):
    submit_button = st.form_submit_button(label='How is my heart looking 💓? ')

if submit_button:

    st.title(body='CardioBot summary')

    st.subheader('Visualize my heart checkup')

    #display 3 visualisations of raw data prior to denoising

    #Make visuals 3x bigger than audio
    col1, col2, col3 = st.columns([1,3,3])

    #Phonocardiogram of raw data
    col1.markdown("Original Audio file of your heartbeat")
    col1.audio(phonocardiogram, format="audio.wav", start_time=0)

    #Spectrogram of raw data
    col2.markdown("Spectrogram of your heartbeat")
    phonocardiogram, fs = librosa.load(phonocardiogram, sr=None)
    phonocardiogram = np.array(phonocardiogram)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(phonocardiogram)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, sr=fs,x_axis='time', y_axis='log')
    plt.xlabel('Time in seconds')
    plt.xlim(left= 0,right= 2)
    col2.pyplot(fig)

    #Oscillogram of raw data
    col3.markdown("Oscillogram of your heartbeat")
    fig_2, ax = plt.subplots()
    img_oscillo = plt.plot(phonocardiogram)
    plt.xlim(left= 0,right= 6000)
    plt.ylim(bottom=-0.5, top=0.5)
    col3.pyplot(fig_2)

    with st.form(key='get_clean_data'):
        submit_button_clean_data = st.form_submit_button(label='Please, clean my phonocardiogram !')
