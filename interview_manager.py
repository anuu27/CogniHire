import sys
from pathlib import Path  # <-- Make sure this import is here
from datetime import datetime, timedelta
import json
import logging

# Import your two modules
from question_generator import QuestionGenerator
from main import process_and_transcribe  # This imports the function from main.py

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterviewManager:
    def __init__(self, resume_path, role):
        if not Path(resume_path).exists():
            logger.error(f"Resume file not found: {resume_path}")
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
            
        self.resume_path = resume_path
        self.role = role
        self.q_generator = QuestionGenerator()
        
        # --- Interview State ---
        self.interview_transcript = []
        self.session_id = f"{Path(resume_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.last_question_asked = None
        self.last_question_type = None  # <-- Correctly initialized
        
        # (Request 3) Time limit
        self.interview_duration_limit = timedelta(minutes=30)
        logger.info(f"Interview manager initialized for {role}. Session: {self.session_id}")
    from threading import Lock


    
    def add_flag(self, issue: str, details: str = ""):
        """
        Record a security/integrity flag for this session.
        Use a lock because Flask may handle requests concurrently.
        """
        entry = {
            "issue": issue,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        with self._lock:
            self.flags.append(entry)
       
        return entry


    def start_interview(self):
        """
        Loads the resume, gets ALL opening remarks + the first question.
        Returns the entire first batch of things for the bot to say.
        """
        logger.info("Setting up interview...")
        # 'opening_remarks' is a list: [intro, gesture]
        opening_remarks = self.q_generator.setup_interview(self.resume_path, self.role, num_role_questions=5)
        
        questions_to_send = list(opening_remarks) # Copy the list

        # Get the first *real* question (which will be role-based)
        first_real_q = self.get_next_question(last_answer="(Interview Started)")
        questions_to_send.append(first_real_q)
        
        # Log all these initial questions
        for q_data in questions_to_send:
            # This helper logs the question AND sets last_question_asked/type
            self.log_new_question(q_data) 
        
        return questions_to_send
        

    def get_next_question(self, last_answer=None):
        """
        Internal helper to get the *next* question from the generator
        based on current state and time.
        """
        current_duration = datetime.now() - self.start_time
        
        # Check for hard time limit
        if current_duration >= self.interview_duration_limit:
            logger.info("Interview time limit reached.")
            return {"type": "end", "question": "Looks like we're at our time limit. Thank you for speaking with me today."}
        
        # Get the next question from the generator
        next_q_data = self.q_generator.get_next_question(
            current_duration=current_duration,
            conversation_history=self.interview_transcript,
            last_answer=last_answer,
            last_question=self.last_question_asked,
            last_question_type=self.last_question_type  # <-- Correctly passed
        )
        
        if next_q_data:
            return next_q_data
        else:
            # Generator returned None, meaning interview is over
            self.save_transcript()
            return {"type": "end", "question": "That's all the questions I have. Thank you for your time!"}

    def process_answer(self, audio_file_path):
        """
        This is the main "live" function.
        1. Transcribes audio.
        2. Gets the next question.
        """
        logger.info(f"Processing audio file: {audio_file_path}")
        user_answer_text = ""
        try:
            if not Path(audio_file_path).exists():
                logger.warning(f"File not found: {audio_file_path}. Treating as empty answer.")
                user_answer_text = "(Audio file not found)"
            else:
                answer_session_id = f"{self.session_id}_{len(self.interview_transcript)}"
                transcript_data = process_and_transcribe(audio_file_path, answer_session_id)
                user_answer_text = transcript_data.get('processed_text', '(No transcription result)')
                logger.info(f"Transcribed Answer: {user_answer_text}")
                
        except Exception as e:
            logger.error(f"Error during audio processing: {e}")
            user_answer_text = "(Error in transcription)"

        # 1. Log the answer to the last question
        self.log_answer_to_last_question(user_answer_text)
        
        # 2. Get the next question
        next_q_data = self.get_next_question(last_answer=user_answer_text)
        
        # 3. Log the new question
        self.log_new_question(next_q_data)
        
        # 4. Return it as a list (for consistency with app.py)
        return [next_q_data]

    def save_transcript(self):
        """Saves the full interview transcript"""
        output_dir = Path("data/interviews")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.session_id}_transcript.json"
        
        full_log = {
            "session_id": self.session_id,
            "role": self.role,
            "resume_file": self.resume_path,
            "candidate_info": self.q_generator.resume_info,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "interview_transcript": self.interview_transcript,
            
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Full interview transcript saved to: {output_file}")
    
    # --- Helper Functions ---

    def log_answer_to_last_question(self, last_answer):
        """Finds the last question in the log and adds the answer to it."""
        if self.last_question_asked and last_answer:
            # Find the last question that hasn't been answered yet
            for i in reversed(range(len(self.interview_transcript))):
                if self.interview_transcript[i]["question"] == self.last_question_asked and self.interview_transcript[i]["answer"] is None:
                    self.interview_transcript[i]["answer"] = last_answer
                    return
            # Fallback if not found (shouldn't happen)
            self.interview_transcript.append({"question": self.last_question_asked, "answer": last_answer})

    def log_new_question(self, new_question_data):
        """Logs a new question and sets it as the 'last_question_asked'."""
        if not new_question_data:
            return
        
        # Get the info from the dictionary
        new_question = new_question_data.get("question")
        new_type = new_question_data.get("type")

        if new_question:
            # Only add if it's not already the last question logged
            if not self.interview_transcript or self.interview_transcript[-1]["question"] != new_question:
                # Also store the type in the log, good for debugging
                self.interview_transcript.append({"question": new_question, "answer": None, "type": new_type})
            
            # Set the last-asked info
            self.last_question_asked = new_question
            self.last_question_type = new_type