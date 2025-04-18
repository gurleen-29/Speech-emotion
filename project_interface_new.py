#import libraries
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
import pyaudio
import tensorflow as tf
import base64
from tensorflow import keras

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("fotor-ai-2024053021131.jpg")

page_background_image = f"""
<style>
[data-testid="stAppViewContainer"]{{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover
}}
"""
st.markdown(page_background_image, unsafe_allow_html=True)

st.title('SER🎵 - SPEECH EMOTION RECOGNITION')
selected = option_menu('HOME',options=['EDA','Interface','About'],default_index=1,icons=['box-fill','clipboard-data','info-circle-fill'],orientation='horizontal')

# model_path = 'C:/Users/rvsin/Desktop/Project/model.h5'
model= tf.keras.models.load_model('model_es.keras')

#INTERFACE
if selected=="Interface":
  st.header("Real Time Audio Emotion Detection")
  option = st.selectbox(
  ':red[**Choose a Sound emotion detection method**] ',
  ('User voice input detection', 'Upload Audio File for Emotion Detection')
  )

  if option == 'User voice input detection':
    st.markdown("<h5 style='color: white;'>User voice input detection</h5>", unsafe_allow_html=True)
    from audio_recorder_streamlit import audio_recorder
    audio_input = audio_recorder(text = 'Click for voice input')

    def save_audio_to_file(audio_file, filename):  
      with open(filename, "wb") as f:
        f.write(audio_file)
    
    def extract_mfcc(file):
      y , sr = librosa.load(file, duration = 3, offset = 0.5)
      mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40).T, axis = 0)
      return mfcc
    
    if audio_input:
      audio_file = st.audio(audio_input, format="audio/wav")
      save_audio_to_file(audio_input, "recorded_audio.wav")
      st.success("Recording complete")
      # st.success("Audio saved successfully!")
      a= extract_mfcc('recorded_audio.wav')
      a = a.reshape(1,40,1)
      ans= model.predict([a])
      st.write(ans)
      # st.write(ans.argmax())
      max_ind = ans.argmax()

      output_dict = {
      0: '**ANGRY**',
      1: '**DISGUST**',
      2: '**FEAR**',
      3: '**HAPPY**',
      4: '**NEUTRAL**',
      5: '**SAD**' 
      }
      
      st.write('OUTPUT : ',output_dict[max_ind])
      # st.markdown("<h1 style='font-size: 24px; font-weight: bold;'>This is bold and larger text using HTML</h1>", unsafe_allow_html=True)

  elif option == 'Upload Audio File for Emotion Detection':
    st.markdown("<h5 style='color: white;'>Upload Audio File for Emotion Detection</h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    # Check if an audio file was uploaded
    if uploaded_file is not None:
      st.audio(uploaded_file, format="audio/mp3")
      st.success("Analyzing the uploaded file...")
      def extract_mfcc(file):
        y , sr = librosa.load(file, duration = 3, offset = 0.5)
        mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40).T, axis = 0)
        return mfcc
      b= extract_mfcc(uploaded_file)
      b = b.reshape(1,40,1)
      ans= model.predict([b])
      # st.write(ans)
      max_ind = ans.argmax()

      output_dict = {
        0: '**ANGRY**',
        1: '**DISGUST**',
        2: '**FEAR**',
        3: '**HAPPY**',
        4: '**NEUTRAL**',
        5: '**SAD**' 
      }
      st.write('OUTPUT : ',output_dict[max_ind])
    else:
      st.write('Please Upload audio file')


#EXPLORATORY DATA ANALYSIS    
if selected=="EDA":
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def load_audio(file_path):
        audio, sr = librosa.load(file_path)
        return audio, sr

    def waveplot(audio, sr):
        def compute_waveform(audio):
          return audio
        # Compute waveform
        waveform_data = compute_waveform(audio)
        # Plot waveform
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(len(waveform_data)) / sr, waveform_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Input Audio Waveform')
        plt.grid(True)
        plt.tight_layout()

        # Display the plot using Streamlit
        st.pyplot()

    def spectrogram(audio, sr):
        # Compute spectrogram
        spectrogram = np.abs(librosa.stft(audio))
        # Plot spectrogram
        plt.figure(figsize=(6, 3))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()

        # Display the plot using Streamlit
        st.pyplot()

    st.title('Audio -Exploratory Data Analysis')

    options = ["Neutral", "Angry", "Disgust","Fear","Happy","Sad"]
    selected_option = st.radio("Select an option:", options)
    st.write("You selected:", selected_option)

    if selected_option == 'Neutral':
       # Load audio file
       uploaded_file = "sample_audio/OAF_back_neutral.wav"
       # Display audio player
       st.write('AUDIO FILE:')
       st.audio(uploaded_file, format="audio/mp3")
       if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)

          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)
    if selected_option == 'Angry':
       # Load audio file
       uploaded_file = "sample_audio/OAF_back_angry.wav"
       # Display audio player
       st.write('AUDIO FILE:')
       st.audio(uploaded_file, format="audio/mp3")
       if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)
         
          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)
    if selected_option == 'Disgust':
       # Load audio file
       uploaded_file = "sample_audio/OAF_back_disgust.wav"
       # Display audio player
       st.write('AUDIO FILE:')
       st.audio(uploaded_file, format="audio/mp3")
       if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)
         
          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)
    if selected_option == 'Fear':
       # Load audio file
       uploaded_file = "sample_audio/OAF_back_fear.wav"
       # Display audio player
       st.write('AUDIO FILE:')
       st.audio(uploaded_file, format="audio/mp3")
       if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)
          
          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)
    if selected_option == 'Happy':
        # Load audio file
        uploaded_file = "sample_audio/OAF_back_happy.wav"
        # Display audio player
        st.write('AUDIO FILE:')
        st.audio(uploaded_file, format="audio/mp3")
        if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)
      
          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)
          
    if selected_option == 'Sad':
        # Load audio file
        uploaded_file = "sample_audio/OAF_back_sad.wav"
        # Display audio player
        st.write('AUDIO FILE:')
        st.audio(uploaded_file, format="audio/mp3")
        if uploaded_file is not None:
          audio, sr = load_audio(uploaded_file)
          
          st.subheader('Waveform')
          waveplot(audio, sr)

          st.subheader('Spectrogram')
          spectrogram(audio, sr)

#ABOUT          
if selected=="About":
    # st.header("About project")
    col1,col2,col3= st.columns([1,1,1])
    with col2:
        st.image('canva_1.png',width=400)

    st.write('**Speech Emotion Recognition (SER) is a system that can identify the emotion of different audio samples. This task is similar to text sentiment analysis, and both also share some applications since they differ only in the modality of the data - text versus audio. Like sentiment analysis, we can use speech emotion recognition to find the emotional range or sentimental value in various audio recordings such as job interviews, caller-agent calls, streaming videos, and songs. Moreover, even music recommendation or classification systems can cluster songs based on their mood and recommend curated playlists to the user.**')
    st.subheader('Applications:')
    st.markdown("""
      <ul>
        <li><h5><strong>Human-Computer Interaction (HCI):</h5></strong>

      **SER can enhance the interaction between humans and computers by allowing systems to adapt to users' emotional states. For example, an automated customer service system can detect frustration in a customer's voice and escalate the call to a human agent.**</li>
        <li><h5><strong>Healthcare:</h5></strong>

      **In mental health applications, SER can assist in diagnosing and monitoring conditions such as depression, anxiety, and schizophrenia by analyzing changes in speech patterns associated with different emotional states.**</li>
        <li><h5><strong>Education:</h5></strong>

      **SER can be used in educational settings to assess students' engagement and emotional states during online lectures or virtual classrooms. It can provide valuable feedback to educators to improve teaching methods and student learning experiences.**</li>
        <li><h5><b>Market Research:</h5></b>

      **SER can be employed in market research to analyze consumers' emotional responses to advertisements, products, or services. This information can help companies tailor their marketing strategies to better resonate with their target audience.**</li>
        <li><h5><b>Security and Law Enforcement:</h5></b>

      **SER can aid in security applications by detecting suspicious behavior or emotional distress in audio recordings, such as emergency calls or surveillance footage. This can assist law enforcement agencies in assessing the severity of a situation and responding accordingly.**</li>
      </ul>
      """, unsafe_allow_html=True)