#!/usr/bin/env python3
"""
Feature Extractor Module
Extracts audio features from utterances for analysis
"""

import numpy as np
import librosa
from scipy import stats
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extract comprehensive audio features from utterances"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.segments_dir = Path("data/segments")
        self.segments_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_features_for_utterances(self, audio_path: str, utterances: List[Dict]) -> List[Dict]:
        """
        Extract features for all utterances in a session
        
        Args:
            audio_path: Path to audio file
            utterances: List of utterance dictionaries
            
        Returns:
            List of utterances with extracted features
        """
        try:
            # Load audio once for efficiency
            audio_array, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            utterances_with_features = []
            for utterance in utterances:
                features = self.extract_utterance_features(audio_path, utterance, audio_array)
                utterances_with_features.append(features)
            
            logger.info(f"Extracted features for {len(utterances_with_features)} utterances")
            return utterances_with_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return []
    
    def extract_utterance_features(self, audio_path: str, utterance: Dict, audio_array: np.ndarray = None) -> Dict:
        """
        Extract comprehensive features for a single utterance
        
        Args:
            audio_path: Path to audio file
            utterance: Utterance dictionary with start/end times
            audio_array: Pre-loaded audio array (optional)
            
        Returns:
            Dictionary with extracted features (preserving original utterance data)
        """
        try:
            # Load audio if not provided
            if audio_array is None:
                audio_array, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            start_sample = int(utterance["start"] * self.sample_rate)
            end_sample = int(utterance["end"] * self.sample_rate)
            utterance_audio = audio_array[start_sample:end_sample]
            
            if len(utterance_audio) == 0:
                return self._get_default_features(utterance)
            
            # Start with the original utterance data
            features = dict(utterance)  # Copy all original data
            
            # IMPROVEMENT 1: Calculate ASR confidence (mean of word confidences)
            features["asr_confidence"] = self._calculate_asr_confidence(utterance)
            
            # Extract basic features (update existing or add new)
            features.update({
                "duration": self._round_float(utterance["end"] - utterance["start"]),
                "word_count": utterance["word_count"]
            })
            
            # Speech rate (words per minute)
            features.update(self._extract_speech_rate(utterance))
            
            # Pause metrics
            features.update(self._extract_pause_metrics(utterance))
            
            # Pitch statistics
            features.update(self._extract_pitch_features(utterance_audio))
            
            # Energy features
            features.update(self._extract_energy_features(utterance_audio))
            
            # Voice quality features
            features.update(self._extract_voice_quality(utterance_audio))
            
            # Spectral features
            features.update(self._extract_spectral_features(utterance_audio))
            
            # Audio quality features
            features.update(self._extract_audio_quality_features(utterance_audio))
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {utterance['utterance_id']}: {str(e)}")
            return self._get_default_features(utterance)
    
    def _calculate_asr_confidence(self, utterance: Dict) -> float:
        """Calculate ASR confidence as mean of word confidence scores"""
        words = utterance.get("words", [])
        if not words:
            return 0.0
        
        confidences = []
        for word in words:
            if word.get("confidence") is not None:
                confidences.append(word["confidence"])
        
        if confidences:
            return self._round_float(np.mean(confidences))
        else:
            return 0.0
    
    def _round_float(self, value: float) -> float:
        """Round float values to 3 decimal places for consistency"""
        if value is None:
            return 0.0
        return round(float(value), 3)
    
    def _extract_speech_rate(self, utterance: Dict) -> Dict:
        """Calculate speech rate metrics"""
        duration = utterance["end"] - utterance["start"]
        word_count = utterance["word_count"]
        
        if duration > 0:
            words_per_second = word_count / duration
            words_per_minute = words_per_second * 60
        else:
            words_per_second = 0
            words_per_minute = 0
        
        return {
            "speech_rate_wps": self._round_float(words_per_second),
            "speech_rate_wpm": self._round_float(words_per_minute),
            "articulation_rate": self._round_float(words_per_second)
        }
    
    def _extract_pause_metrics(self, utterance: Dict) -> Dict:
        """Extract pause-related metrics from word timings"""
        words = utterance.get("words", [])
        
        if len(words) < 2:
            return {
                "pause_count": 0,
                "pause_rate": 0,
                "mean_pause_duration": 0,
                "std_pause_duration": 0,
                "total_pause_time": 0,
                "speaking_time_ratio": 1.0
            }
        
        # Calculate pauses between words
        pauses = []
        for i in range(1, len(words)):
            pause_duration = words[i]["start"] - words[i-1]["end"]
            if pause_duration > 0.1:  # Only count pauses > 100ms
                pauses.append(pause_duration)
        
        if pauses:
            mean_pause = np.mean(pauses)
            std_pause = np.std(pauses)
            total_pause_time = sum(pauses)
        else:
            mean_pause = 0
            std_pause = 0
            total_pause_time = 0
        
        utterance_duration = utterance["end"] - utterance["start"]
        speaking_time = utterance_duration - total_pause_time
        
        return {
            "pause_count": len(pauses),
            "pause_rate": self._round_float(len(pauses) / max(len(words) - 1, 1)),
            "mean_pause_duration": self._round_float(mean_pause),
            "std_pause_duration": self._round_float(std_pause),
            "total_pause_time": self._round_float(total_pause_time),
            "speaking_time_ratio": self._round_float(speaking_time / utterance_duration) if utterance_duration > 0 else 0
        }
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """Extract pitch-related features using librosa"""
        try:
            # Extract pitch using pyin (robust pitch tracking)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=50, 
                fmax=400, 
                sr=self.sample_rate,
                frame_length=2048
            )
            
            # Get only voiced frames
            voiced_f0 = f0[voiced_flag]
            
            if len(voiced_f0) > 0:
                pitch_mean = np.mean(voiced_f0)
                pitch_std = np.std(voiced_f0)
                pitch_range = np.max(voiced_f0) - np.min(voiced_f0)
                pitch_variance = np.var(voiced_f0)
            else:
                pitch_mean = pitch_std = pitch_range = pitch_variance = 0
            
            return {
                "pitch_mean": self._round_float(pitch_mean),
                "pitch_std": self._round_float(pitch_std),
                "pitch_range": self._round_float(pitch_range),
                "pitch_variance": self._round_float(pitch_variance),
                "voiced_frames_ratio": self._round_float(np.mean(voiced_flag))
            }
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {str(e)}")
            return self._get_default_pitch_features()
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict:
        """Extract energy-related features"""
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
            rms_values = rms[0]
            
            # Energy envelope features
            energy_mean = np.mean(rms_values)
            energy_std = np.std(rms_values)
            energy_max = np.max(rms_values)
            energy_min = np.min(rms_values)
            energy_dynamic_range = energy_max - energy_min if energy_max > 0 else 0
            
            # Zero-crossing rate (articulation indicator)
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
            zcr_mean = np.mean(zcr)
            
            return {
                "rms_energy_mean": self._round_float(energy_mean),
                "rms_energy_std": self._round_float(energy_std),
                "rms_energy_max": self._round_float(energy_max),
                "rms_energy_min": self._round_float(energy_min),
                "energy_dynamic_range": self._round_float(energy_dynamic_range),
                "zero_crossing_rate": self._round_float(zcr_mean)
            }
            
        except Exception as e:
            logger.warning(f"Energy extraction failed: {str(e)}")
            return self._get_default_energy_features()
    
    def _extract_voice_quality(self, audio: np.ndarray) -> Dict:
        """Extract voice quality features"""
        try:
            # Harmonic-to-noise ratio (roughness measure)
            harmonic, percussive = librosa.effects.hpss(audio)
            hnr = np.sum(harmonic**2) / np.sum(percussive**2) if np.sum(percussive**2) > 0 else 0
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            centroid_mean = np.mean(spectral_centroid)
            centroid_std = np.std(spectral_centroid)
            
            return {
                "harmonic_noise_ratio": self._round_float(hnr),
                "spectral_centroid_mean": self._round_float(centroid_mean),
                "spectral_centroid_std": self._round_float(centroid_std)
            }
            
        except Exception as e:
            logger.warning(f"Voice quality extraction failed: {str(e)}")
            return {
                "harmonic_noise_ratio": 0,
                "spectral_centroid_mean": 0,
                "spectral_centroid_std": 0
            }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Extract spectral features"""
        try:
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            rolloff_mean = np.mean(rolloff)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            bandwidth_mean = np.mean(bandwidth)
            
            return {
                "mfcc_1_mean": self._round_float(mfcc_mean[0]),
                "mfcc_2_mean": self._round_float(mfcc_mean[1]),
                "mfcc_1_std": self._round_float(mfcc_std[0]),
                "spectral_rolloff": self._round_float(rolloff_mean),
                "spectral_bandwidth": self._round_float(bandwidth_mean)
            }
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {str(e)}")
            return self._get_default_spectral_features()
    
    def _extract_audio_quality_features(self, audio: np.ndarray) -> Dict:
        """Extract audio quality features"""
        try:
            # Signal-to-noise ratio approximation
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            snr_approx = rms_mean / rms_std if rms_std > 0 else 0
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            
            return {
                "snr_approximation": self._round_float(snr_approx),
                "clipping_ratio": self._round_float(clipping_ratio)
            }
            
        except Exception as e:
            logger.warning(f"Audio quality extraction failed: {str(e)}")
            return {
                "snr_approximation": 0,
                "clipping_ratio": 0
            }
    
    def _get_default_features(self, utterance: Dict) -> Dict:
        """Return default features when extraction fails (preserving original data)"""
        features = dict(utterance)  # Copy all original utterance data
        
        # Add/update with default feature values
        default_features = {
            "duration": self._round_float(utterance["end"] - utterance["start"]),
            "word_count": utterance["word_count"],
            "asr_confidence": 0.0,
            "speech_rate_wps": 0,
            "speech_rate_wpm": 0,
            "articulation_rate": 0,
            "pause_count": 0,
            "pause_rate": 0,
            "mean_pause_duration": 0,
            "std_pause_duration": 0,
            "total_pause_time": 0,
            "speaking_time_ratio": 0
        }
        
        # Update with default pitch features
        default_features.update(self._get_default_pitch_features())
        default_features.update(self._get_default_energy_features())
        default_features.update(self._get_default_spectral_features())
        
        features.update(default_features)
        return features
    
    def _get_default_pitch_features(self) -> Dict:
        return {
            "pitch_mean": 0,
            "pitch_std": 0,
            "pitch_range": 0,
            "pitch_variance": 0,
            "voiced_frames_ratio": 0
        }
    
    def _get_default_energy_features(self) -> Dict:
        return {
            "rms_energy_mean": 0,
            "rms_energy_std": 0,
            "rms_energy_max": 0,
            "rms_energy_min": 0,
            "energy_dynamic_range": 0,
            "zero_crossing_rate": 0
        }
    
    def _get_default_spectral_features(self) -> Dict:
        return {
            "mfcc_1_mean": 0,
            "mfcc_2_mean": 0,
            "mfcc_1_std": 0,
            "spectral_rolloff": 0,
            "spectral_bandwidth": 0,
            "harmonic_noise_ratio": 0,
            "spectral_centroid_mean": 0,
            "spectral_centroid_std": 0,
            "snr_approximation": 0,
            "clipping_ratio": 0
        }
    
    def save_features_to_json(self, utterances_with_features: List[Dict], session_id: str):
        """Save features to JSON file"""
        output_file = self.segments_dir / f"{session_id}_features.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(utterances_with_features, f, indent=2, ensure_ascii=False)
        logger.info(f"Features saved to: {output_file}")
        return output_file
    
    def save_features_to_parquet(self, utterances_with_features: List[Dict], session_id: str):
        """Save features to Parquet file"""
        df = pd.DataFrame(utterances_with_features)
        output_file = self.segments_dir / f"{session_id}_features.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"Features saved to: {output_file}")
        return output_file
    
    def save_features_to_csv(self, utterances_with_features: List[Dict], session_id: str):
        """Save features to CSV file"""
        df = pd.DataFrame(utterances_with_features)
        output_file = self.segments_dir / f"{session_id}_features.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Features saved to: {output_file}")
        return output_file

# Convenience function
def extract_features(audio_path: str, utterances: List[Dict], session_id: str, 
                   output_formats: List[str] = ['json', 'csv']) -> List[Dict]:
    """
    Convenience function to extract and save features
    
    Args:
        audio_path: Path to audio file
        utterances: List of utterance dictionaries
        session_id: Session identifier
        output_formats: List of formats to save ('json', 'csv', 'parquet')
        
    Returns:
        List of utterances with features
    """
    extractor = AudioFeatureExtractor()
    utterances_with_features = extractor.extract_features_for_utterances(audio_path, utterances)
    
    # Save in requested formats
    if 'json' in output_formats:
        extractor.save_features_to_json(utterances_with_features, session_id)
    if 'csv' in output_formats:
        extractor.save_features_to_csv(utterances_with_features, session_id)
    if 'parquet' in output_formats:
        extractor.save_features_to_parquet(utterances_with_features, session_id)
    
    return utterances_with_features

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Feature Extractor')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('transcript_file', help='Path to transcript JSON file')
    parser.add_argument('--session', help='Session ID')
    parser.add_argument('--formats', nargs='+', default=['json', 'csv'], 
                       choices=['json', 'csv', 'parquet'],
                       help='Output formats')
    
    args = parser.parse_args()
    
    # Load transcript file
    with open(args.transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    utterances = transcript_data.get('utterances', [])
    
    if not utterances:
        print("No utterances found in transcript file")
        exit(1)
    
    # Extract features
    features = extract_features(args.audio_path, utterances, args.session, args.formats)
    print(f"Extracted features for {len(features)} utterances")
    print(f"Files saved to: data/segments/")