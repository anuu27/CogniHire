#!/usr/bin/env python3
"""
Main Pipeline Script
Complete audio processing, transcription, and feature extraction
"""

import argparse
from pathlib import Path
from ingest import AudioIngestor
from asr_wrapper import ASRWrapper
import traceback

def process_and_transcribe(audio_path, session_id=None, model_size="base"):
    """Complete pipeline: ingest audio → transcribe → extract features"""
    
    # Step 1: Process audio
    print("🔊 Processing audio...")
    ingestor = AudioIngestor()
    processed_audio, metadata_file, metadata = ingestor.process_audio(audio_path)
    
    # Step 2: Transcribe
    print("🎤 Transcribing audio...")
    asr = ASRWrapper(model_size=model_size)
    
    if session_id is None:
        session_id = Path(audio_path).stem
    
    transcript = asr.transcribe_file(str(processed_audio), session_id)
    
    # Step 3: Extract features
    print("📊 Extracting features...")
    from feature_extractor import extract_features
    utterances_with_features = extract_features(
        str(processed_audio), 
        transcript['utterances'], 
        session_id, 
        ['json', 'csv']
    )
    
    # Update transcript with features
    transcript['utterances'] = utterances_with_features
    
    # Step 4: Display results
    print("\n" + "="*50)
    print("📝 TRANSCRIPTION COMPLETE")
    print("="*50)
    print(f"Session: {transcript['session_id']}")
    print(f"Duration: {transcript['duration']}s")
    print(f"Language: {transcript['language']}")
    print(f"Word count: {transcript['word_count']}")
    print(f"Utterances: {transcript['utterance_count']}")
    print(f"\nProcessed Text:\n{transcript['processed_text']}")
    print("\nUtterances with features:")
    
    # Safe display - check for required keys
    for utterance in transcript['utterances'][:3]:
        if 'utterance_id' in utterance and 'start' in utterance and 'end' in utterance:
            print(f"  {utterance['utterance_id']}: {utterance['start']:.2f}-{utterance['end']:.2f}s")
            print(f"    Words: {utterance.get('word_count', 'N/A')}")
            if 'speech_rate_wpm' in utterance:
                print(f"    Speech Rate: {utterance.get('speech_rate_wpm', 'N/A')} WPM")
            if 'pitch_mean' in utterance:
                print(f"    Pitch Mean: {utterance.get('pitch_mean', 'N/A')} Hz")
        else:
            print(f"  Utterance missing required keys: {utterance.keys()}")
    
    return transcript

def process_folder(folder_path, model_size="base"):
    """Process all audio files in a folder"""
    audio_folder = Path(folder_path)
    if not audio_folder.exists():
        raise FileNotFoundError(f"Audio folder not found: {folder_path}")
    
    # Supported audio formats
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.aac']:
        audio_files.extend(audio_folder.glob(ext))
    
    if not audio_files:
        print("❌ No audio files found in the folder")
        return []
    
    print(f"🎵 Found {len(audio_files)} audio files")
    results = []
    successful_files = 0
    
    for audio_file in audio_files:
        try:
            print(f"\n🔊 Processing: {audio_file.name}")
            session_id = audio_file.stem
            transcript = process_and_transcribe(str(audio_file), session_id, model_size)
            results.append(transcript)
            successful_files += 1
            print(f"✅ Successfully processed: {audio_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {audio_file.name}: {e}")
            traceback.print_exc()
            continue
    
    return results, successful_files

def main():
    parser = argparse.ArgumentParser(description='Audio Processing & Transcription Pipeline')
    parser.add_argument('input_path', help='Path to input audio file or folder')
    parser.add_argument('--session', help='Session ID (optional for single file)')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all files in folder (batch mode)')
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input_path)
        
        if args.batch or input_path.is_dir():
            # Process folder
            if not input_path.exists():
                print(f"❌ Folder not found: {args.input_path}")
                return
            
            results, successful_files = process_folder(args.input_path, args.model)
            print(f"\n✅ Batch processing complete! Successfully processed {successful_files} files.")
            
        else:
            # Process single file
            if not input_path.exists():
                print(f"❌ File not found: {args.input_path}")
                return
            
            transcript = process_and_transcribe(args.input_path, args.session, args.model)
            
            # Save a simple summary
            summary_file = f"transcript_{transcript['session_id']}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Session: {transcript['session_id']}\n")
                f.write(f"Audio: {args.input_path}\n")
                f.write(f"Text: {transcript['processed_text']}\n")
            
            print(f"\n💾 Summary saved: {summary_file}")
            print("📁 Full transcripts in: data/transcripts/")
            print("📊 Features in: data/segments/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()