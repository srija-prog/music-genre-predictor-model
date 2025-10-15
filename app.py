import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pickle import UnpicklingError
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import io
import tempfile

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with stronger dark-mode support
st.markdown("""
    <style>
    /* Base app containers */
    .stApp, .main, .block-container {
        transition: background 0.3s ease, color 0.3s ease;
    }

    /* Light theme defaults (kept subtle) */
    .genre-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    /* Dark-mode overrides */
    @media (prefers-color-scheme: dark) {
        .stApp, .main, .block-container {
            background: linear-gradient(180deg, #071021 0%, #0f1724 60%, #0b1220 100%) !important;
            color: #d1d9e6 !important;
        }

        /* Upload area */
        .upload-box {
            border: 1px dashed rgba(255,255,255,0.06) !important;
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            backdrop-filter: blur(6px);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #0b1220 0%, #102132 100%) !important;
            color: #e6eef8 !important;
            border-radius: 20px !important;
            padding: 10px 28px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.04) !important;
            box-shadow: 0 8px 30px rgba(2,6,23,0.6) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 40px rgba(2,6,23,0.7) !important;
        }

        /* Genre card (darker, translucent) */
        .genre-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
            border: 1px solid rgba(255,255,255,0.03) !important;
            color: #e6eef8 !important;
            box-shadow: 0 6px 18px rgba(2,6,23,0.6) !important;
        }

        /* Metric card (accented darker) */
        .metric-card {
            background: linear-gradient(180deg,#071824 0%, #0b1220 100%) !important;
            border-radius: 12px !important;
            padding: 18px !important;
            color: #e6eef8 !important;
            border: 1px solid rgba(255,255,255,0.03) !important;
            box-shadow: 0 8px 30px rgba(2,6,23,0.65) !important;
        }

        /* Headings & text */
        h1, h2, h3, p, label, span {
            color: #e6eef8 !important;
        }

        /* Plot backgrounds */
        .stPlotlyChart, .element-container, .stImage {
            background: transparent !important;
        }

        /* Tweak sidebar appearance */
        .css-1d391kg, .css-1lcbmhc {
            background: transparent !important;
        }
    }

    </style>
""", unsafe_allow_html=True)

# Helper function to extract MFCC features
def extract_mfcc(audio_data, sr, duration=30, n_mfcc=40):
    try:
        # Limit to duration if needed
        max_len = duration * sr
        if len(audio_data) > max_len:
            audio_data = audio_data[:max_len]
        
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Load or train model
@st.cache_resource
def load_model():
    # Check if pre-trained model exists
    model_files = ['genre_classifier.pkl', 'scaler.pkl', 'label_encoder.pkl']
    if all(os.path.exists(p) for p in model_files):
        try:
            # Primary: try pickle
            with open('genre_classifier.pkl', 'rb') as f:
                clf = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            return clf, scaler, label_encoder
        except (UnpicklingError, EOFError, AttributeError, ImportError, Exception) as e:
            st.error(f"Failed to load saved model files using pickle: {e}")
            # Try joblib as a fallback if available (sometimes models saved differently)
            try:
                import joblib
                clf = joblib.load('genre_classifier.pkl')
                scaler = joblib.load('scaler.pkl')
                label_encoder = joblib.load('label_encoder.pkl')
                return clf, scaler, label_encoder
            except Exception as je:
                st.warning("Saved model files appear corrupted or incompatible. Removing them so you can retrain.")
                for p in model_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                return None, None, None
    else:
        st.warning("⚠️ Pre-trained model not found. Please train the model first using the sidebar.")
        return None, None, None

# Main header
st.markdown("<h1 style='text-align: center; font-size: 60px;'>🎵 Music Genre Classifier AI 🎸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 20px;'>Upload your music and let AI identify the genre!</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/music.png", width=100)
    st.title("🎛️ Control Panel")
    
    page = st.radio("Navigation", ["🎵 Predict Genre", "📊 Model Info", "🎓 Train Model"])
    
    st.markdown("---")
    st.markdown("### 🎼 Supported Genres")
    genres = ["Blues", "Classical", "Country", "Disco", "Hip-Hop", 
              "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    for genre in genres:
        st.markdown(f"• {genre}")

# Page: Predict Genre
if page == "🎵 Predict Genre":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Music File")
        uploaded_file = st.file_uploader("Choose an audio file (WAV format recommended)", 
                                        type=['wav', 'mp3', 'ogg', 'flac'],
                                        help="Upload a music file to classify its genre")
        
        if uploaded_file is not None:
            # Load model
            clf, scaler, label_encoder = load_model()
            
            if clf is not None:
                with st.spinner('🎵 Analyzing your music... This might take a moment!'):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Load audio
                        audio_data, sr = librosa.load(tmp_path, duration=30)
                        
                        # Display audio player
                        st.audio(uploaded_file, format='audio/wav')
                        
                        # Extract features
                        mfcc_features = extract_mfcc(audio_data, sr)
                        
                        if mfcc_features is not None:
                            # Scale and predict
                            features_scaled = scaler.transform([mfcc_features])
                            prediction = clf.predict(features_scaled)[0]
                            probabilities = clf.predict_proba(features_scaled)[0]
                            
                            # Get genre name
                            predicted_genre = label_encoder.classes_[prediction]
                            confidence = probabilities[prediction] * 100
                            
                            # Display results
                            st.success("✅ Analysis Complete!")
                            
                            st.markdown(f"""
                                <div class='genre-card'>
                                    <h2 style='text-align: center; color: #667eea;'>🎸 Detected Genre</h2>
                                    <h1 style='text-align: center; color: #764ba2; font-size: 48px;'>{predicted_genre.upper()}</h1>
                                    <p style='text-align: center; font-size: 24px; color: #666;'>Confidence: {confidence:.2f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability distribution
                            st.markdown("### 📊 Genre Probability Distribution")
                            
                            # Create DataFrame for plotting
                            import pandas as pd
                            prob_df = pd.DataFrame({
                                'Genre': label_encoder.classes_,
                                'Probability': probabilities * 100
                            }).sort_values('Probability', ascending=True)
                            
                            # Create horizontal bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = plt.cm.viridis(np.linspace(0, 1, len(prob_df)))
                            bars = ax.barh(prob_df['Genre'], prob_df['Probability'], color=colors)
                            ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Genre', fontsize=12, fontweight='bold')
                            ax.set_title('Genre Classification Probabilities', fontsize=14, fontweight='bold')
                            
                            # Highlight the predicted genre
                            for i, bar in enumerate(bars):
                                if prob_df.iloc[i]['Genre'] == predicted_genre:
                                    bar.set_color('#764ba2')
                                    bar.set_linewidth(3)
                                    bar.set_edgecolor('gold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # MFCC Visualization
                            st.markdown("### 🎼 Audio Features Visualization (MFCC)")
                            mfcc_full = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
                            
                            fig2, ax2 = plt.subplots(figsize=(12, 6))
                            img = librosa.display.specshow(mfcc_full, x_axis='time', sr=sr, ax=ax2, cmap='coolwarm')
                            ax2.set_title('Mel-Frequency Cepstral Coefficients (MFCC)', fontsize=14, fontweight='bold')
                            ax2.set_ylabel('MFCC Coefficients', fontsize=12)
                            ax2.set_xlabel('Time (s)', fontsize=12)
                            plt.colorbar(img, ax=ax2, format='%+2.0f dB')
                            plt.tight_layout()
                            st.pyplot(fig2)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **For Best Results:**
        - Use WAV format files
        - Upload at least 30 seconds of audio
        - Ensure good audio quality
        - Avoid heavily mixed tracks
        """)
        
        st.markdown("### 🎯 How It Works")
        st.success("""
        1. **Upload** your music file
        2. **Extract** MFCC features
        3. **Analyze** with Random Forest AI
        4. **Get** instant genre prediction!
        """)

# Page: Model Info
elif page == "📊 Model Info":
    st.markdown("## 📊 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3>🤖 Algorithm</h3>
                <p style='font-size: 24px; font-weight: bold;'>Random Forest</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3>🎵 Features</h3>
                <p style='font-size: 24px; font-weight: bold;'>40 MFCC Coefficients</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <h3>🎸 Genres</h3>
                <p style='font-size: 24px; font-weight: bold;'>10 Categories</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🧠 About MFCC (Mel-Frequency Cepstral Coefficients)")
    st.write("""
    MFCC is a powerful audio feature extraction technique that:
    - Represents the short-term power spectrum of sound
    - Mimics human auditory system perception
    - Captures timbral characteristics of music
    - Provides compact representation of audio signals
    """)
    
    st.markdown("### 🎯 Model Architecture")
    st.write("""
    **Random Forest Classifier:**
    - 100 decision trees (estimators)
    - Trained on GTZAN dataset
    - StandardScaler normalization
    - Achieves ~67% accuracy on test data
    """)

# Page: Train Model
elif page == "🎓 Train Model":
    st.markdown("## 🎓 Train Your Own Model")
    
    st.info("📂 Place your dataset in a folder named 'dataset/' with subfolders for each genre containing .wav files")
    
    dataset_path = st.text_input("Dataset Path", "dataset/")
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
    with col2:
        test_size = st.slider("Test Split %", 10, 30, 20) / 100
    
    if st.button("🚀 Train Model", type="primary"):
        if not os.path.exists(dataset_path):
            st.error("❌ Dataset path not found!")
        else:
            with st.spinner("🎵 Training model... This may take a few minutes!"):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.metrics import classification_report, accuracy_score
                    
                    # Load data
                    features = []
                    labels = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_files = []
                    for genre in os.listdir(dataset_path):
                        genre_path = os.path.join(dataset_path, genre)
                        if os.path.isdir(genre_path):
                            for file in os.listdir(genre_path):
                                if file.lower().endswith('.wav'):
                                    all_files.append((os.path.join(genre_path, file), genre))
                    
                    total_files = len(all_files)
                    
                    for idx, (file_path, genre) in enumerate(all_files):
                        status_text.text(f"Processing: {os.path.basename(file_path)}")
                        
                        try:
                            y, sr = librosa.load(file_path, duration=30)
                            mfcc = extract_mfcc(y, sr)
                            if mfcc is not None:
                                features.append(mfcc)
                                labels.append(genre)
                        except:
                            pass
                        
                        progress_bar.progress((idx + 1) / total_files)
                    
                    # Train model
                    X = np.array(features)
                    y = np.array(labels)
                    
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_encoded, test_size=test_size, random_state=42
                    )
                    
                    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    clf.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Save models
                    with open('genre_classifier.pkl', 'wb') as f:
                        pickle.dump(clf, f)
                    with open('scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    with open('label_encoder.pkl', 'wb') as f:
                        pickle.dump(label_encoder, f)
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                    
                    st.markdown("### 📈 Classification Report")
                    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
                    st.text(report)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Made with ❤️ using Streamlit | Powered by AI 🤖</p>
        <p>🎵 Music Genre Classifier v1.0</p>
    </div>
""", unsafe_allow_html=True)