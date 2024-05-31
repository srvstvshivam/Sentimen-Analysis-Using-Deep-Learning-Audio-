import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
# Define emotion categories
emotion_categories = ['angry', 'happy', 'sad', 'fear','neutral','PleasantSurprise']

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def load_data(data_path):
    X, y = [], []
    # Assume data_path is structured by folders named by categories
    for category in emotion_categories:
        category_path = os.path.join(data_path, category)
        for file in os.listdir(category_path):
            if file.lower().endswith('.wav'):
                feature = extract_features(os.path.join(category_path, file))
                X.append(feature)
                y.append(emotion_categories.index(category))
    return np.array(X), np.array(y)

from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_path, save_directory, model_name="emotion_mod_lstm", epochs=450, batch_size=32):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = to_categorical(y_train, num_classes=len(emotion_categories))
    y_test = to_categorical(y_test, num_classes=len(emotion_categories))

    model = build_model(X_train.shape[1], len(emotion_categories))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # Save the model with a full file path including a file extension
    model_save_path = os.path.join(save_directory, model_name + ".h5")  # Ensure the extension is .h5
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    data_directory = 'C:/Users/nikhi/Desktop/archive/TESS Toronto emotional speech set data'
    model_save_directory = 'C:/Users/nikhi/Desktop/project/speech-emotion-webapp'  # Ensure this directory exists or create it
    trained_model = train_model(data_directory, model_save_directory)