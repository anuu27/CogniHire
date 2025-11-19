


import google.generativeai as genai
import json
import re
from PyPDF2 import PdfReader
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os
from datetime import timedelta
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()   
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.error("FATAL: GEMINI_API_KEY environment variable not set.")

class QuestionGenerator:
    def __init__(self, index_file="data/faiss_index.bin"):
        self.questions = []
        self.index = self._load_index(index_file)
        self.sbert_model = self._load_sbert_model()
        self.gemini_model = genai.GenerativeModel("models/gemini-2.5-flash") 
        self.role = "Unknown"
        self.resume_info = {}
        self.role_questions = []
        self.followup_questions = []
        self.resume_questions_asked = [] 
        self.wrap_up_asked = False
        self.wrap_up_start_time = timedelta(minutes=29)

    def _load_questions(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f: return json.load(f)
        except: return []

    def _load_index(self, file_path):
        try: return faiss.read_index(file_path)
        except: return None

    def _load_sbert_model(self):
        try: return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except: return None

    def _extract_text_from_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() for page in reader.pages])
        except: return ""

    def _safe_json_extract(self, text):
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            return json.loads(match.group(0)) if match else None
        except: return None

    def _extract_resume_info(self, resume_text):
        prompt = f"""
        Extract resume info. Return ONLY valid JSON: {{ "name": "", "skills": [], "projects": [] }}
        Resume: {resume_text}
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            return self._safe_json_extract(response.text) or {"name": "Unknown", "skills": [], "projects": []}
        except: return {"name": "Unknown", "skills": [], "projects": []}

    def _ask_followup_questions(self, main_question, answer, conversation_history):
        prompt = f"""
        You are a casual, friendly interviewer. 
        Main Question: "{main_question}"
        Candidate Answer: "{answer}"
        
        Task: If the answer is too short or vague, ask ONE simple, short follow-up question to clarify.
        Keep it conversational. DO NOT use complex jargon.
        If the answer is fine, return empty list.
        
        Return JSON: {{ "followups": ["..."] }}
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            parsed = self._safe_json_extract(response.text)
            return parsed.get("followups", [])[:1] if parsed else []
        except: return []


    def _search_role_questions(self, role, top_k=5, seed=None):
        """
        Load role-specific questions from data/role_questions/<role>.json
        and return up to `top_k` randomly sampled questions (no duplicates).
        If the file contains fewer than top_k questions, return a shuffled copy of all questions.

        Optional:
        - seed: if provided (int), sampling will be deterministic (useful for tests).
        """
        # normalize role -> safe filename key
        if not role:
            logger.info("[QG] _search_role_questions called with empty role")
            return []

        # convert to lowercase, replace non-alphanumeric with underscore, collapse underscores
        key = re.sub(r'[^a-z0-9]+', '_', role.lower().strip())
        key = re.sub(r'_+', '_', key).strip('_')
        filename = f"{key}.json"
        file_path = Path("data/role_questions") / filename

        logger.info(f"[QG] Looking for role file for role='{role}' -> '{file_path}'")

        if not file_path.exists():
            logger.warning(f"[QG] Role questions not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # Normalize payload into list[str]
            questions_list = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        questions_list.append(item)
                    elif isinstance(item, dict) and "question" in item:
                        questions_list.append(item["question"])
            elif isinstance(payload, dict):
                if "questions" in payload and isinstance(payload["questions"], list):
                    for item in payload["questions"]:
                        if isinstance(item, str):
                            questions_list.append(item)
                        elif isinstance(item, dict) and "question" in item:
                            questions_list.append(item["question"])
                else:
                    # fallback: collect string values
                    for v in payload.values():
                        if isinstance(v, str):
                            questions_list.append(v)

            total = len(questions_list)
            if total == 0:
                logger.warning(f"[QG] No questions found in {file_path}")
                return []

            # optional deterministic sampling for tests
            rng = random.Random(seed) if seed is not None else random

            # If there are at least top_k questions, sample top_k unique questions
            if total >= top_k:
                sampled = rng.sample(questions_list, k=top_k)
            else:
                # fewer than top_k: return shuffled list (so order is random each time)
                sampled = list(questions_list)
                rng.shuffle(sampled)

            logger.info(f"[QG] Loaded {len(sampled)} role questions from {file_path} (total in file: {total})")
            if len(sampled) > 0:
                logger.info(f"[QG] Sample questions (first 3): {sampled[:3]}")
            return sampled

        except Exception as e:
            logger.exception(f"[QG] Failed to load role questions for {role}: {e}")
            return []


    def _generate_next_resume_question(self, conversation_history):
        if not self.resume_questions_asked:
            instruction = "Ask a simple, easy starter question about one of their projects. Keep it under 15 words."
        else:
            instruction = f"""
            Ask a simple question about a DIFFERENT skill or project from their resume.
            Keep it conversational and short (under 20 words).
            Avoid these topics: {self.resume_questions_asked}
            """

        prompt = f"""
        Role: {self.role}
        Resume: {json.dumps(self.resume_info)}
        Task: {instruction}
        Return JSON: {{ "question": "..." }}
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            parsed = self._safe_json_extract(response.text)
            if parsed and "question" in parsed:
                self.resume_questions_asked.append(parsed["question"])
                return parsed["question"]
            return "Tell me about your favorite project."
        except: return "What is a project you enjoyed?"

    def setup_interview(self, pdf_path, role, num_role_questions=5):
        self.role = role
        text = self._extract_text_from_pdf(pdf_path)
        self.resume_info = self._extract_resume_info(text) if text else {}
        self.role_questions = self._search_role_questions(role, top_k=num_role_questions)
        self.role_questions = self._search_role_questions(role, top_k=num_role_questions)
        logger.info(f"[QG] setup_interview role='{role}' loaded {len(self.role_questions)} role_questions")

        
        # --- MODIFIED INTRO (Single Item) ---
        return [{"type": "intro", "question": "Hi, nice to meet you. Let's get started."}]

    def get_next_question(self, current_duration, conversation_history, last_answer=None, last_question=None, last_question_type=None) -> dict:
        if current_duration >= self.wrap_up_start_time or self.wrap_up_asked:
            self.wrap_up_asked = True
            return None 

        if self.role_questions:
            return {"type": "role", "question": self.role_questions.pop(0)}
        
        if last_answer and "skipped" in last_answer.lower():
            return {"type": "resume", "question": self._generate_next_resume_question(conversation_history)}

        if last_question_type == "follow-up":
            return {"type": "resume", "question": self._generate_next_resume_question(conversation_history)}

        if last_answer and last_question_type == "resume":
            followups = self._ask_followup_questions(last_question, last_answer, conversation_history)
            if followups: return {"type": "follow-up", "question": followups[0]}

        return {"type": "resume", "question": self._generate_next_resume_question(conversation_history)}