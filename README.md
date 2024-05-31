# Sentimen Analysis Using Deep Learning(Audio)
# Speech Emotion Recognition

Web-application based on ML model for recognition of emotion for selected audio file.

## Description
The project you described involves Speech Emotion Recognition (SER) using several well-known datasets and advanced machine learning techniques. Let's break down the key components and methodologies used in your project:

### Datasets
1. **Crowd-sourced Emotional Multimodal Actors Dataset (Crema-D)**: This dataset includes audio-visual recordings of actors expressing a range of emotions.
2. **Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)**: This dataset contains 24 actors performing two lexically-matched statements in a range of emotions and two emotional songs.
3. **Surrey Audio-Visual Expressed Emotion (Savee)**: This dataset features male actors expressing various emotions, providing both audio and visual data.
4. **Toronto Emotional Speech Set (Tess)**: This dataset includes recordings of two actresses saying a set of 200 target words in seven different emotions.

### Digital Signal Processing and SER
Digital Signal Processing (DSP) is crucial for extracting meaningful features from speech signals, which are then used for emotion recognition. 

### Key Processes in SER:
1. **Feature Selection**: Identifying and extracting discriminative features from speech signals that can accurately represent the emotional state of the speaker.
2. **Classification**: Using machine learning models to classify the extracted features into different emotion categories.

### Common Features and Techniques:
1. **Mel-Scale Filter Bank Speech Spectrogram**:
   - **Spectrogram**: A 2D visual representation of the spectrum of frequencies in a sound signal as they vary with time, widely used in CNNs.
   - **Mel Scale**: A perceptual scale of pitches judged by listeners to be equal in distance from one another. It approximates human hearing more closely than a linear scale.

2. **Mel-Frequency Cepstral Coefficients (MFCCs)**:
   - MFCCs are derived from the Fourier transform of a signal and represent the short-term power spectrum of sound.
   - They capture the timbral aspects of the audio signal, which are essential for distinguishing different emotions.

### Deep Learning Techniques:
- **Convolutional Neural Networks (CNNs)**: Effective for processing 2D data like spectrograms to extract salient features.
- **Pre-trained CNNs**: Transfer learning using models like VGG, DenseNet, or Alex-Net can improve performance by leveraging features learned from large datasets.

### Your Approach:
- **Combined Model**: Utilizing a pre-trained DenseNet for mel-spectrograms and a CNN for MFCCs. This hybrid approach aims to leverage the strengths of both types of features for more accurate emotion recognition.

This project represents a sophisticated approach to SER, combining state-of-the-art datasets and machine learning techniques to tackle the challenging task of emotion recognition from speech signals.## Installation

It is recommended to use the provided `requirements.txt` file to set your virtual environment.

Sure, here’s a concise version of the setup instructions:

### Setup and Run the Application

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/srvstvshivam/Sentimen-Analysis-Using-Deep-Learning-Audio-
    ```

2. **Navigate to the Directory**:
    ```sh
    cd Sentimen-Analysis-Using-Deep-Learning-Audio
    ```

3. **Create a Virtual Environment**:
    ```sh
    python -m virtualenv your_venv
    ```

4. **Activate the Virtual Environment**:
    - On Windows:
      ```sh
      your_venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      source your_venv/bin/activate
      ```

5. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

6. **Run the Streamlit App**:
    ```sh
    streamlit run app.py
    ```

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
