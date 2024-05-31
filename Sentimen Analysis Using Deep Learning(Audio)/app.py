import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile

# Define emotion categories
emotion_categories = ['angry', 'happy', 'sad', 'fear','neutral','PleasantSurprise']

# Function to load the model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    Xdb = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    return mfccs, Xdb, sr

# Function to plot custom styled audio waveform
def plot_custom_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor('#d1d1e0')
    plt.title("Wave-form")
    librosa.display.waveshow(y, sr=sr)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    plt.gca().axes.spines["bottom"].set_visible(False)
    plt.gca().axes.set_facecolor('#d1d1e0')
    return fig

# Function to predict emotion
def predict_emotion(audio_path, model):
    mfccs, Xdb, sr = extract_features(audio_path)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)  # Reshape for prediction
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    return emotion_categories[predicted_index], mfccs, Xdb, sr

# Streamlit app setup
def main():
    st.title("Sentimen Analysis Using Deep Learning(Audio)")
    model_path = "C:/Users/offic/Downloads/project/Sentimen Analysis Using Deep Learning(Audio)/emotion_mod_lstm.h5"
    model = load_model(model_path)

    st.sidebar.header("Upload Audio")
    audio_file = st.sidebar.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file.seek(0)  # Go to the start of the file
            # Display custom styled waveform
            fig = plot_custom_waveform(tmp_file.name)
            st.write(fig)

            predicted_emotion, mfccs, Xdb, sr = predict_emotion(tmp_file.name, model)
            
            st.write("## Analysis:")
            # Display the analysis graphs
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(10, 2))
                fig.set_facecolor('#d1d1e0')
                plt.title("MFCCs(Mel-Frequency Cepstral Coefficients)")
                librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                plt.gca().axes.spines["bottom"].set_visible(False)
                plt.gca().axes.set_facecolor('#d1d1e0')
                st.pyplot(fig)

            with col2:
                fig2 = plt.figure(figsize=(10, 2))
                fig2.set_facecolor('#d1d1e0')
                plt.title("Mel-log-spectrogram")
                librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                plt.gca().axes.spines["bottom"].set_visible(False)
                plt.gca().axes.set_facecolor('#d1d1e0')
                st.pyplot(fig2)
            
            st.write("## Prediction")
            st.write("Predicted Emotion:", predicted_emotion)
if __name__ == "__main__":
    main()