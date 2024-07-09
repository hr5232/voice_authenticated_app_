from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from featureextraction import extract_features
from audio_processing import preprocess_audio
import warnings
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

MODEL_PATH = "C:/Users/Dell/Desktop/voice-auth-app/trained_models/"
INPUT_FOLDER = "C:/Users/Dell/Desktop/voice-auth-app/input/"
OUTPUT_FOLDER = "C:/Users/Dell/Desktop/voice-auth-app/output/"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signin', methods=['POST'])
def upload_audio():
    if 'files' not in request.files or 'speaker' not in request.form:
        return jsonify({'error': 'No files or speaker name provided'}), 400

    files = request.files.getlist('files')
    speaker = request.form['speaker']
    
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400

    features = np.asarray(())

    for file in files:
        if file.filename == '':
            continue
        filepath = os.path.join(INPUT_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the audio
        preprocessed_filepath = os.path.join(OUTPUT_FOLDER, f'preprocessed_{file.filename}')
        preprocess_audio(filepath, preprocessed_filepath)

        # Perform feature extraction
        sr, audio = read(preprocessed_filepath)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    if features.size == 0:
        return jsonify({'error': 'No valid audio files provided'}), 400

    # Train the GMM model
    gmm = GMM(n_components=5, covariance_type='diag', n_init=3)
    gmm.fit(features)

    # Save the model
    model_filename = os.path.join(MODEL_PATH, f"{speaker}.gmm")
    cPickle.dump(gmm, open(model_filename, 'wb'))
    return jsonify({'message': f'Model for {speaker} trained successfully'}), 200

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def recognize_speaker():
    if 'file' not in request.files or 'test_speaker' not in request.form:
        return jsonify({'error': 'No file or speaker name provided'}), 400

    file = request.files['file']
    test_speaker = request.form['test_speaker']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(INPUT_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess the audio
    preprocessed_filepath = os.path.join(OUTPUT_FOLDER, f'preprocessed_{file.filename}')
    preprocess_audio(filepath, preprocessed_filepath)

    sr, audio = read(preprocessed_filepath)
    vector = extract_features(audio, sr)

    # Dynamically load the models
    gmm_files = [os.path.join(MODEL_PATH, fname) for fname in os.listdir(MODEL_PATH) if fname.endswith('.gmm')]
    models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

    if len(models) == 0:
        return jsonify({'error': 'No models found'}), 400

    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]
        try:
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        except:
            log_likelihood[i] = -np.inf  # Handle potential score errors

    if np.all(log_likelihood == -np.inf):
        return jsonify({'error': 'No valid scores obtained from models'}), 400

    winner = np.argmax(log_likelihood)
    recognized_speaker = speakers[winner]

    if recognized_speaker == test_speaker:
        return jsonify({'message': 'Speaker verified successfully', 'recognized_speaker': recognized_speaker}), 200
    else:
        return jsonify({'error': 'Speaker verification failed', 'recognized_speaker': recognized_speaker}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)