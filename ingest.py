#!/usr/bin/env python3
"""
Audio Ingestion Module
Standardizes audio files to normalized .wav (16kHz, mono)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import librosa 
import soundfile as sf
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioIngestor:
    def __init__(self, target_sr=16000, target_channels=1):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.output_dir = Path("audio_analysis_ingest")  # FIXED: Changed from data/sampled_audio
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_audio(self, file_path):
        """Load audio file using librosa"""
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def get_audio_metadata(self, file_path, audio, sr, original_channels):
        """Extract audio metadata"""
        duration = len(audio) / sr
        return {
            "original_file": str(file_path),
            "duration_seconds": round(duration, 2),
            "original_sample_rate": sr,
            "target_sample_rate": self.target_sr,
            "original_channels": original_channels,
            "target_channels": self.target_channels,
            "normalized": True,
            "silence_trimmed": True
        }
    
    def convert_to_mono(self, audio):
        """Convert multi-channel audio to mono"""
        if len(audio.shape) > 1:
            logger.info(f"Converting {audio.shape[0]} channels to mono")
            audio = librosa.to_mono(audio)
        return audio
    
    def resample_audio(self, audio, original_sr):
        """Resample audio to target sample rate"""
        if original_sr != self.target_sr:
            logger.info(f"Resampling from {original_sr}Hz to {self.target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        return audio
    
    def normalize_amplitude(self, audio, target_level=-23):
        """Normalize audio amplitude to target LUFS level"""
        try:
            # Calculate RMS (simplified normalization)
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                # Simple normalization to avoid clipping
                max_val = np.max(np.abs(audio))
                if max_val > 0.9:  # If too loud, scale down
                    scaling_factor = 0.9 / max_val
                    audio = audio * scaling_factor
                    logger.info(f"Reduced amplitude to prevent clipping")
                
                # Target RMS for consistent volume
                target_rms = 10**(target_level / 20)
                current_rms = np.sqrt(np.mean(audio**2))
                if current_rms > 0:
                    audio = audio * (target_rms / current_rms)
            
            return audio
        except Exception as e:
            logger.warning(f"Normalization failed: {str(e)}. Using original audio.")
            return audio
    
    def trim_silence(self, audio, sr, top_db=20):
        """Trim leading and trailing silence"""
        try:
            # Trim silence using librosa
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            
            if len(audio_trimmed) < len(audio):
                logger.info(f"Trimmed {len(audio) - len(audio_trimmed)} samples of silence")
                return audio_trimmed
            else:
                logger.info("No significant silence found to trim")
                return audio
        except Exception as e:
            logger.warning(f"Silence trimming failed: {str(e)}. Using original audio.")
            return audio
    
    def save_audio_and_metadata(self, audio, metadata, output_stem):
        """Save processed audio and metadata"""
        # Save audio file
        audio_filename = self.output_dir / f"{output_stem}.wav"
        sf.write(audio_filename, audio, self.target_sr, subtype='PCM_16')
        logger.info(f"Saved processed audio: {audio_filename}")
        
        # Save metadata
        metadata_filename = self.output_dir / f"{output_stem}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_filename}")
        
        return audio_filename, metadata_filename
    
    def process_audio(self, input_file):
        """Main processing pipeline"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Processing: {input_path}")
        
        # Load audio
        audio, sr = self.load_audio(input_path)
        original_channels = audio.shape[0] if len(audio.shape) > 1 else 1
        
        # Process audio
        audio = self.convert_to_mono(audio)
        audio = self.resample_audio(audio, sr)
        audio = self.normalize_amplitude(audio)
        audio = self.trim_silence(audio, self.target_sr)
        
        # Get metadata
        metadata = self.get_audio_metadata(input_path, audio, sr, original_channels)
        metadata["final_duration_seconds"] = round(len(audio) / self.target_sr, 2)
        
        # Generate output filename
        output_stem = f"{input_path.stem}_processed"
        
        # Save results
        audio_file, metadata_file = self.save_audio_and_metadata(audio, metadata, output_stem)
        
        return audio_file, metadata_file, metadata

def main():
    parser = argparse.ArgumentParser(description='Standardize audio ingestion')
    parser.add_argument('input_file', help='Path to input audio file (wav/mp3/m4a)')
    parser.add_argument('--output-dir', default='audio_analysis_ingest', 
                       help='Output directory for processed files (default: audio_analysis_ingest)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    
    args = parser.parse_args()
    
    # Initialize ingestor
    ingestor = AudioIngestor(target_sr=args.sample_rate)
    
    try:
        # Process audio
        audio_file, metadata_file, metadata = ingestor.process_audio(args.input_file)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Input: {args.input_file}")
        print(f"Output Audio: {audio_file}")
        print(f"Output Metadata: {metadata_file}")
        print(f"Duration: {metadata['final_duration_seconds']}s")
        print(f"Sample Rate: {metadata['target_sample_rate']}Hz")
        print(f"Channels: {metadata['target_channels']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()