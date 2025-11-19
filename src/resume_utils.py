import re
import json
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# You can later plug this into interview_manager.py


# ---------- 1️⃣ Basic keyword extraction ----------
def extract_keywords(resume_text, top_n=10):
    """Return top resume keywords based on frequency (excluding stopwords)."""
    text = resume_text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    stop = {
        "with","have","that","from","this","about","which","their","been","were","your","also",
        "into","such","using","used","able","will","team","work","project","projects","experience",
        "developer","engineer","management","skills","ability","knowledge","worked","like"
    }
    filtered = [w for w in words if w not in stop]
    top = Counter(filtered).most_common(top_n)
    return [w for w, _ in top]


# ---------- 2️⃣ TF-IDF based keyphrase extraction ----------
def extract_phrases(resume_text, top_n=5):
    """Use TF-IDF to get top unique noun-like phrases."""
    sentences = re.split(r'[.!?]', resume_text)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vec.fit_transform(sentences)
    scores = zip(vec.get_feature_names_out(), X.sum(axis=0).A1)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [phrase for phrase, _ in sorted_scores]


# ---------- 3️⃣ Optional summarization ----------
def summarize_resume(resume_text, max_len=400):
    """Rough summarization — pick top sentences containing strong action verbs."""
    action_verbs = ["led","developed","designed","managed","built","created","implemented","analyzed","coordinated"]
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    key_lines = [l for l in lines if any(v in l.lower() for v in action_verbs)]
    summary = " ".join(key_lines)[:max_len]
    return summary


# ---------- 4️⃣ Generate structured resume context ----------
def get_resume_context(resume_text):
    """
    Return structured info usable by the interviewer pipeline.
    {
      'summary': ...,
      'keywords': [...],
      'phrases': [...]
    }
    """
    context = {
        "summary": summarize_resume(resume_text),
        "keywords": extract_keywords(resume_text, top_n=8),
        "phrases": extract_phrases(resume_text, top_n=5)
    }
    return context


# ---------- 5️⃣ Example demo ----------
if __name__ == "__main__":
    resume = """
    Experienced Data Scientist with 6 years of experience in machine learning, deep learning,
    and AI-powered analytics. Led a 4-person data team to deploy predictive models using Python,
    TensorFlow, and SQL. Skilled in data visualization, NLP, and cloud deployment.
    """
    ctx = get_resume_context(resume)
    print(json.dumps(ctx, indent=2))
