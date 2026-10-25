from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def generate_synthetic_acoustic_distribution(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # 46 features: 40 MFCCs + [pitch_mean, pitch_var, zcr_mean, spectral_centroid_mean, spectral_rolloff_mean, rms_var]
    
    # REAL VOICE ACOUSTIC DISTRIBUTION (Human speech characteristics)
    # High micro-jitter, natural pitch variance, natural ambient spectral centroid and roll-off
    real_mfcc = np.random.normal(loc=-15.0, scale=35.0, size=(n_samples, 40))
    real_pitch_mean = np.random.normal(loc=185.0, scale=45.0, size=(n_samples, 1)) # 120-250 Hz typical human pitch
    real_pitch_var = np.random.normal(loc=950.0, scale=300.0, size=(n_samples, 1))  # Natural human vocal jitter & pitch contour
    real_zcr_mean = np.random.normal(loc=0.085, scale=0.03, size=(n_samples, 1))    # Natural consonant fricatives
    real_spectral_centroid = np.random.normal(loc=2200.0, scale=450.0, size=(n_samples, 1))
    real_spectral_rolloff = np.random.normal(loc=4400.0, scale=700.0, size=(n_samples, 1))
    real_rms_var = np.random.normal(loc=0.012, scale=0.005, size=(n_samples, 1))    # Natural dynamic breath & emphasis modulation
    
    x_real = np.hstack([
        real_mfcc, real_pitch_mean, real_pitch_var, real_zcr_mean,
        real_spectral_centroid, real_spectral_rolloff, real_rms_var
    ])
    y_real = np.zeros(n_samples, dtype=int)
    
    # AI / SYNTHETIC VOICE ACOUSTIC DISTRIBUTION (Neural Vocoder HiFi-GAN / ElevenLabs / Bark)
    # Abnormally smooth pitch contour (low pitch variance), lower vocoder spectral roll-off (8-10kHz mel cutoff),
    # lower dynamic RMS variation (compressed amplitude envelope), characteristic vocoder MFCC spectral tilt
    fake_mfcc = np.random.normal(loc=-8.0, scale=28.0, size=(n_samples, 40))
    # Synthetic vocoder pitch has distinct lower variance or periodic phase artifacts
    fake_pitch_mean = np.random.normal(loc=175.0, scale=25.0, size=(n_samples, 1))
    fake_pitch_var = np.random.normal(loc=280.0, scale=90.0, size=(n_samples, 1))   # Significantly lower natural pitch micro-jitter
    fake_zcr_mean = np.random.normal(loc=0.055, scale=0.02, size=(n_samples, 1))
    fake_spectral_centroid = np.random.normal(loc=1750.0, scale=300.0, size=(n_samples, 1))
    fake_spectral_rolloff = np.random.normal(loc=3100.0, scale=450.0, size=(n_samples, 1)) # Truncated vocoder high frequency
    fake_rms_var = np.random.normal(loc=0.0035, scale=0.0015, size=(n_samples, 1)) # Compressed dynamic range
    
    x_fake = np.hstack([
        fake_mfcc, fake_pitch_mean, fake_pitch_var, fake_zcr_mean,
        fake_spectral_centroid, fake_spectral_rolloff, fake_rms_var
    ])
    y_fake = np.ones(n_samples, dtype=int)
    
    x = np.vstack([x_real, x_fake])
    y = np.concatenate([y_real, y_fake])
    return x, y

def main():
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "voice"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "voice_model.pkl"
    scaler_path = model_dir / "voice_scaler.pkl"
    
    print("Generating acoustic training dataset for deepfake & neural voice detection...")
    x, y = generate_synthetic_acoustic_distribution(n_samples=3000)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training RandomForest acoustic classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42)
    model.fit(x_train, y_train)
    
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n==========================================")
    print(f"Voice Detector Model Accuracy: {acc * 100:.2f}%")
    print(f"==========================================")
    print(classification_report(y_test, preds, target_names=["Real Voice", "AI Voice"]))
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved voice model to: {model_path}")
    print(f"Saved voice scaler to: {scaler_path}")

if __name__ == "__main__":
    main()
