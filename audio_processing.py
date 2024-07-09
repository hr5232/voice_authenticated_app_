import glob
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfilt
import os

# Function to normalize audio using librosa
def normalize_audio(y):
    rms = np.sqrt(np.mean(y**2))
    target_rms = 0.1  # target RMS level (e.g., -20 dBFS)
    y = y * (target_rms / rms)
    return y

# Function to remove noise
def remove_noise(y, sr):
    y = nr.reduce_noise(y=y, sr=sr)
    return y

# Function to trim silence
def trim_silence(y, top_db=20):
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y

# Function to equalize audio
def equalize_audio(y, sr):
    y = librosa.effects.preemphasis(y)
    return y

# Function to filter audio (high-pass filter to remove low-frequency noise)
def highpass_filter(y, sr, cutoff=100):
    sos = butter(10, cutoff, btype='highpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    return y

# Function to preprocess audio
def preprocess_audio(input_file, output_file):
    y, sr = librosa.load(input_file, sr=None)
    
    # Apply preprocessing steps
    y = remove_noise(y, sr)
    y = trim_silence(y)
    y = equalize_audio(y, sr)
    y = highpass_filter(y, sr)
    y = normalize_audio(y)
    
    # Save the preprocessed audio to a new file
    sf.write(output_file, y, sr)