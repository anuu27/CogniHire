#!/usr/bin/env python3
"""
ASR Wrapper Module using faster-whisper with Voice Activity Detection (VAD)
Replaces whisper with faster-whisper for Windows-friendly operation.

Interface kept similar to previous version:
- ASRWrapper.transcribe_file(audio_path, session_id=None, language=None, task="transcribe", verbose=True, enable_vad=True)
Returns a structured dict containing text, segments, words, utterances, etc.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
import numpy as np
import time
import webrtcvad
import struct

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CACHE TO PREVENT RELOADING CRASHES ---
GLOBAL_MODEL = None

# Try to import faster-whisper; if unavailable, set a fallback stub
HAVE_ASR = True
try:
    from faster_whisper import WhisperModel
except Exception as e:
    logger.warning(f"faster-whisper import failed: {e}. ASR functionality will be stubbed.")
    HAVE_ASR = False


class VADSegmenter:
    """Voice Activity Detection using webrtcvad"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000, frame_duration: int = 30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
    def detect_voice_activity(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        audio_int16 = (audio * 32767).astype(np.int16)
        segments = []
        current_segment = None
        min_silence_frames = 5
        min_voice_frames = 3
        silence_frames = 0
        
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
            frame_bytes = struct.pack('<' + 'h' * len(frame), *frame)
            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                logger.warning(f"VAD is_speech check failed: {e}")
                continue

            current_time = i / self.sample_rate
            if is_speech:
                silence_frames = 0
                if current_segment is None:
                    current_segment = [current_time, current_time + self.frame_duration / 1000]
                else:
                    current_segment[1] = current_time + self.frame_duration / 1000
            else:
                silence_frames += 1
                if current_segment is not None and silence_frames >= min_silence_frames:
                    if (current_segment[1] - current_segment[0]) >= (min_voice_frames * self.frame_duration / 1000):
                        segments.append(tuple(current_segment))
                    current_segment = None
                    silence_frames = 0
        
        if current_segment is not None:
            if (current_segment[1] - current_segment[0]) >= (min_voice_frames * self.frame_duration / 1000):
                segments.append(tuple(current_segment))
        
        merged_segments = self._merge_close_segments(segments, gap_threshold=0.5)
        logger.info(f"VAD detected {len(merged_segments)} voice segments")
        return merged_segments
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]], gap_threshold: float = 0.5) -> List[Tuple[float, float]]:
        if not segments:
            return []
        merged = []
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start - current_end <= gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged


class ASRWrapper:
    def __init__(self, model_size: str = "base", vad_aggressiveness: int = 2, device: str = "cpu", compute_type: str = "int8_float32"):
        """
        model_size: tiny/base/small/medium/large
        device: 'cpu' or 'cuda:0' (if you have GPU)
        compute_type: compute type string for faster-whisper; 'int8_float32' is useful for CPU memory savings
        """
        self.model_size = model_size
        self.model = None
        self.vad_segmenter = VADSegmenter(aggressiveness=vad_aggressiveness)
        self.transcripts_dir = Path("data/transcripts")
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.compute_type = compute_type

    def load_model(self):
        """Load Whisper model (lazy loading, cached globally)"""
        global GLOBAL_MODEL
        if not HAVE_ASR:
            raise RuntimeError("ASR backend (faster-whisper) not available in this environment.")
        
        if self.model is None:
            if GLOBAL_MODEL is None:
                logger.info(f"Loading faster-whisper model: {self.model_size} (device={self.device}, compute_type={self.compute_type})")
                try:
                    # This will download model into cache if missing
                    GLOBAL_MODEL = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                    logger.info("Model loaded successfully (Global Cache)")
                except Exception as e:
                    logger.error(f"Failed to load faster-whisper model: {e}")
                    raise
            else:
                logger.info("Using cached global faster-whisper model")
            self.model = GLOBAL_MODEL
        return self.model

    def transcribe_file(self, audio_path: str, session_id: str = None, 
                       language: str = None, task: str = "transcribe",
                       verbose: bool = True, enable_vad: bool = True, beam_size: int = 5) -> Dict:
        """
        Transcribe entire audio file with optional VAD segmentation.
        Returns structured result with utterances mapped to VAD segments.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if session_id is None:
            session_id = self._generate_session_id()
        logger.info(f"Transcribing: {audio_path} (Session: {session_id})")
        
        # Load audio for VAD processing using librosa for consistent sr
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Librosa failed to load audio for VAD: {e}")
            audio = None
            sr = 16000
        
        voice_segments = []
        if enable_vad and audio is not None:
            voice_segments = self.vad_segmenter.detect_voice_activity(audio)
        
        # If ASR backend unavailable, return a stubbed empty result structure
        if not HAVE_ASR:
            logger.warning("ASR backend unavailable; returning empty transcription structure.")
            empty_result = {
                "text": "",
                "segments": [],
                "language": language or "unknown",
                "duration": 0
            }
            processed_result = self._process_transcription_with_utterances(empty_result, voice_segments, session_id, audio_path)
            processed_result["processing_time_seconds"] = 0.0
            processed_result["vad_segments_detected"] = len(voice_segments)
            processed_result["vad_enabled"] = enable_vad
            return processed_result

        # Load the model and run transcription
        model = self.load_model()
        start_time = time.time()
        # faster-whisper returns (segments, info)
        segments_iter, info = model.transcribe(audio_path, beam_size=beam_size, word_timestamps=True, language=language)
        # segments_iter is an iterator/generator — convert to list
        segments_list = list(segments_iter)
        processing_time = time.time() - start_time

        # Build result in a whisper-like dict shape
        result = {
            "text": " ".join([seg.text for seg in segments_list]).strip(),
            "segments": [],
            "language": info.language if hasattr(info, "language") else (language or "unknown"),
            "duration": getattr(info, "duration", None) or None
        }

        # Each seg in faster-whisper typically has: start, end, text, words (if word_timestamps True)
        for seg in segments_list:
            seg_dict = {
                "id": None,
                "seek": None,
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": getattr(seg, "text", ""),
                "words": []
            }
            # words may be available as seg.words or seg.word_timestamps depending on version
            words_attr = getattr(seg, "words", None) or getattr(seg, "word_timestamps", None)
            if words_attr:
                for w in words_attr:
                    # w may be a simple tuple or object with attributes
                    try:
                        w_text = getattr(w, "text", None) or (w[2] if len(w) > 2 else None) or (w[0] if len(w) > 0 else "")
                        w_start = float(getattr(w, "start", None) or w[0])
                        w_end = float(getattr(w, "end", None) or w[1])
                    except Exception:
                        # fallback generic mapping
                        try:
                            w_text = w[2]
                            w_start = float(w[0])
                            w_end = float(w[1])
                        except Exception:
                            continue
                    seg_dict["words"].append({"word": w_text, "start": round(w_start, 3), "end": round(w_end, 3)})
            result["segments"].append(seg_dict)

        # Store raw transcription
        raw_transcript_path = self._store_raw_transcript(result, session_id, audio_path)

        # Process into utterances mapped to VAD segments
        processed_result = self._process_transcription_with_utterances(result, voice_segments, session_id, audio_path)
        processed_result["processing_time_seconds"] = round(processing_time, 2)
        processed_result["vad_segments_detected"] = len(voice_segments)
        processed_result["vad_enabled"] = enable_vad

        logger.info(f"Transcription completed in {processing_time:.1f}s for session: {session_id}")
        logger.info(f"Created {len(processed_result['utterances'])} utterances from {len(voice_segments)} VAD segments")
        return processed_result

    def _process_transcription_with_utterances(self, result: Dict, voice_segments: List[Tuple[float, float]], session_id: str, audio_source: str) -> Dict:
        # Extract words with timestamps (normalise to same structure you used before)
        words = []
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                w_text = word.get("word", "").strip()
                w_start = round(word.get("start", 0), 3)
                w_end = round(word.get("end", 0), 3)
                words.append({
                    "text": w_text,
                    "start": w_start,
                    "end": w_end,
                    "confidence": None
                })

        raw_text = result.get("text", "").strip()
        processed_text = self._post_process_text(raw_text)
        utterances = self._create_utterances_from_segments(words, voice_segments)

        # Calculate actual duration
        try:
            import librosa
            y, sr = librosa.load(audio_source, sr=None)
            actual_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            logger.warning(f"Could not get actual duration from file: {e}")
            if words:
                actual_duration = max(word["end"] for word in words)
            else:
                actual_duration = result.get("duration", 0) or 0

        for utterance in utterances:
            utterance["features"] = {
                "speech_rate_wpm": None,
                "speech_rate_wps": None,
                # ... keep the same placeholders as before
            }

        structured_result = {
            "session_id": session_id,
            "audio_source": audio_source,
            "raw_text": raw_text,
            "processed_text": processed_text,
            "words": words,
            "utterances": utterances,
            "language": result.get("language", "unknown"),
            "duration": round(actual_duration, 2),
            "model_used": self.model_size,
            "timestamp": datetime.now().isoformat(),
            "segment_count": len(utterances),
            "word_count": len(words),
            "utterance_count": len(utterances)
        }

        processed_filename = self.transcripts_dir / f"{session_id}_processed.json"
        try:
            with open(processed_filename, 'w', encoding='utf-8') as f:
                json.dump(structured_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save processed transcript: {e}")

        logger.info(f"Processed transcript with utterances saved: {processed_filename}")
        return structured_result

    def _create_utterances_from_segments(self, words: List[Dict], voice_segments: List[Tuple[float, float]]) -> List[Dict]:
        utterances = []
        for i, (segment_start, segment_end) in enumerate(voice_segments):
            segment_words = [
                word for word in words 
                if word["start"] >= segment_start and word["end"] <= segment_end
            ]
            if segment_words:
                utterance_start = min(word["start"] for word in segment_words)
                utterance_end = max(word["end"] for word in segment_words)
                transcript = " ".join(word["text"] for word in segment_words)
                processed_transcript = self._post_process_text(transcript)
                utterance = {
                    "utterance_id": f"u{i+1:03d}",
                    "start": round(utterance_start, 3),
                    "end": round(utterance_end, 3),
                    "duration": round(utterance_end - utterance_start, 3),
                    "transcript": processed_transcript,
                    "word_count": len(segment_words),
                    "words": segment_words,
                    "vad_segment_start": round(segment_start, 3),
                    "vad_segment_end": round(segment_end, 3),
                }
                utterances.append(utterance)
        return utterances

    def _post_process_text(self, text: str) -> str:
        if not text:
            return text
        processed = text
        processed = re.sub(r'\s+([.,!?;:])', r'\1', processed)
        processed = re.sub(r'([.,!?;:])(\w)', r'\1 \2', processed)
        sentences = re.split(r'([.!?])\s+', processed)
        if len(sentences) > 1:
            processed_tmp = ''
            for i in range(0, len(sentences)-1, 2):
                sentence = sentences[i].strip()
                punctuation = sentences[i+1] if i+1 < len(sentences) else ''
                if sentence:
                    processed_tmp += sentence[0].upper() + sentence[1:] + punctuation + ' '
            processed = processed_tmp.strip()
        filler_words = [
            r'\bum\b', r'\buh\b', r'\bah\b', r'\buhm\b', r'\ber\b',
            r'\blike\b', r'\byou know\b', r'\by\'know\b',
            r'\bactually\b', r'\bbasically\b', r'\bliterally\b',
            r'\bhonestly\b', r'\banyway\b', r'\bso\s+$'
        ]
        for filler in filler_words:
            processed = re.sub(filler, '', processed, flags=re.IGNORECASE)
        processed = re.sub(r'\s+', ' ', processed).strip()
        if processed and processed[0].islower():
            processed = processed[0].upper() + processed[1:]
        return processed

    # placeholders for other methods kept as pass to preserve API
    def transcribe_segment(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                          session_id: str = None, language: str = None) -> Dict:
        raise NotImplementedError("transcribe_segment is not implemented in this wrapper.")

    def transcribe_large_file_safely(self, audio_path: str, session_id: str = None,
                                   chunk_minutes: int = 10, language: str = None) -> Dict:
        raise NotImplementedError("transcribe_large_file_safely is not implemented in this wrapper.")

    def batch_transcribe(self, audio_directory: str, file_pattern: str = "*.wav") -> List[Dict]:
        raise NotImplementedError("batch_transcribe is not implemented in this wrapper.")

    def _generate_session_id(self) -> str:
        return f"s{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _store_raw_transcript(self, result: Dict, session_id: str, audio_source: str) -> str:
        raw_filename = self.transcripts_dir / f"{session_id}_raw.json"
        try:
            with open(raw_filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return str(raw_filename)
        except Exception as e:
            logger.warning(f"Failed to store raw transcript: {e}")
            return ""

# Convenience functions
def transcribe_file(audio_path: str, session_id: str = None, model_size: str = "base", **kwargs) -> Dict:
    asr = ASRWrapper(model_size=model_size)
    return asr.transcribe_file(audio_path, session_id, **kwargs)

def transcribe_with_vad(audio_path: str, session_id: str = None, model_size: str = "base", 
                       vad_aggressiveness: int = 2, **kwargs) -> Dict:
    asr = ASRWrapper(model_size=model_size, vad_aggressiveness=vad_aggressiveness)
    return asr.transcribe_file(audio_path, session_id, **kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ASR Transcription Wrapper with VAD (faster-whisper)')
    parser.add_argument('audio_path', help='Path to audio file or directory')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--session', help='Session ID (optional)')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch')
    parser.add_argument('--language', help='Language code (e.g., en, es, fr)')
    parser.add_argument('--no-vad', action='store_true', help='Disable VAD segmentation')
    parser.add_argument('--vad-aggressiveness', type=int, default=2, choices=[0, 1, 2, 3],
                       help='VAD aggressiveness 0-3 (default: 2)')
    args = parser.parse_args()

    asr = ASRWrapper(model_size=args.model, vad_aggressiveness=args.vad_aggressiveness)
    if args.batch:
        results = asr.batch_transcribe(args.audio_path)
        print(f"Batch processing complete. Processed {len(results)} files.")
    else:
        result = asr.transcribe_file(args.audio_path, args.session, language=args.language, enable_vad=not args.no_vad)
        print(f"\nTranscription complete for session: {result['session_id']}")
        print(f"VAD segments detected: {result['vad_segments_detected']}")
        print(f"Utterances created: {result['utterance_count']}")
        print(f"Language: {result['language']}")
        print(f"\nUtterances:")
        for utterance in result['utterances'][:5]:
            print(f"  {utterance['utterance_id']}: {utterance['start']:.2f}-{utterance['end']:.2f}s")
            print(f"    Text: {utterance['transcript']}")
