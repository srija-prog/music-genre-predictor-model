# 🎵 AI Music Genre Detector

## Overview
The **AI Music Genre Detector** is a machine learning project that predicts the genre of a song based on its audio features. Using **MFCC (Mel-Frequency Cepstral Coefficients)** extracted from audio files, the system identifies patterns in the music and classifies it into genres like Pop, Rock, Jazz, Classical, Metal, and more. The project includes a trained model, preprocessing components, and an interactive **Streamlit app** for real-time predictions.

---

## Features
- Supports **.mp3** and **.wav** audio files  
- Extracts **MFCC audio features** using `librosa`  
- Uses a **Random Forest classifier** for accurate genre prediction  
- Interactive **Streamlit app** with:  
  - Audio playback  
  - Waveform visualization  
  - MFCC heatmap visualization  
  - Confidence bar chart for all genres  
- Color-coded predictions for a visually appealing experience  

---

## Repository Structure
├── dataset/ # Organized audio files by genre
├── music_genre_classifier.ipynb # Jupyter notebook with training & evaluation
├── genre_classifier.pkl # Trained ML model
├── scaler.pkl # Scaler for feature normalization
├── label_encoder.pkl # Label encoder for genre labels
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
└── README.md


---

## Installation
1. Clone the repository:

git clone <repository_url>
cd <repository_folder>
for creating a virtual env

python -m venv musicenv
# Windows
musicenv\Scripts\activate
# macOS/Linux
source musicenv/bin/activate
install requiremnts:_
pip install -r requirements.txt

Dataset

The project uses the GTZAN Music Genre Dataset
 with 10 genres, 100 audio clips per genre, each 30 seconds long. Files are in .wav format.

##Technologies Used:-

Python, NumPy, Pandas
Scikit-learn (Random Forest)
Librosa (Audio feature extraction)
Matplotlib & Seaborn (Visualization)
Streamlit (Web app)
Joblib (Model saving/loading)


