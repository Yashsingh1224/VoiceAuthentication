from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import os
from scipy.spatial.distance import cosine

app = FastAPI()
EMBEDDINGS_DIR = "Embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def preprocess_audio(file, sr=16000):
    audio, _ = librosa.load(file, sr=sr)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    return audio

def extract_features(audio, sr=16000, max_pad_len=100):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    pad_width = max_pad_len - features.shape[1]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_pad_len]
    return np.mean(features, axis=1)

@app.post("/register-user")
async def register_user(username: str = Form(...), file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...)):
    files = [file1, file2, file3]
    embeddings = []

    for file in files:
        with open(file.filename, 'wb') as f:
            f.write(await file.read())
        audio = preprocess_audio(file.filename)
        features = extract_features(audio)
        embeddings.append(features)
        os.remove(file.filename)

    avg_embedding = np.mean(np.array(embeddings), axis=0)
    np.save(os.path.join(EMBEDDINGS_DIR, f"{username}.npy"), avg_embedding)
    return JSONResponse({"message": f"User {username} registered successfully."})

@app.post("/verify-voice")
async def verify_voice(username: str = Form(...), file: UploadFile = File(...)):
    user_embedding_path = os.path.join(EMBEDDINGS_DIR, f"{username}.npy")
    if not os.path.exists(user_embedding_path):
        return JSONResponse({"match": False, "message": "User not found."})

    with open(file.filename, 'wb') as f:
        f.write(await file.read())
    audio = preprocess_audio(file.filename)
    test_embedding = extract_features(audio)
    os.remove(file.filename)

    saved_embedding = np.load(user_embedding_path)
    similarity = 1 - cosine(saved_embedding, test_embedding)

    match = similarity > 0.75
    return JSONResponse({"match": match, "similarity": similarity})
