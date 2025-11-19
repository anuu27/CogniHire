import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
QUESTION_FILE = "data/questions.json"
INDEX_FILE = "data/faiss_index.bin"

# Load model
print("📥 Loading Sentence-BERT model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========== 1️⃣ Load Questions ==========
def load_questions():
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return questions


# ========== 2️⃣ Build FAISS Index ==========
def build_index(questions):
    texts = [q["question"] for q in questions]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ Built FAISS index with {len(texts)} questions")
    return index, embeddings


# ========== 3️⃣ Load Existing Index ==========
def load_index():
    index = faiss.read_index(INDEX_FILE)
    questions = load_questions()
    return questions, index


# ========== 4️⃣ Core Search (with role + resume context) ==========
def search(query, questions, index, role=None, resume=None, job_desc=None, top_k=5):
    """
    Enhanced retrieval:
    - query: main search query (e.g., 'leadership' or 'project management')
    - role: filter questions belonging to this role
    - resume: text string from candidate's resume (optional)
    - job_desc: job description string (optional)
    - top_k: number of questions to return
    """
    # Combine resume / JD context to bias query
    context = ""
    if resume:
        context += f" Resume: {resume[:400]}."  # truncate long resumes
    if job_desc:
        context += f" Job Description: {job_desc[:400]}."
    
    full_query = f"{query}. {context}"
    q_vec = model.encode([full_query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, top_k * 3)  # retrieve more, then filter

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(questions): continue

        q = questions[idx]
        if role and q["role"].lower() != role.lower():
            continue  # skip non-role questions
        results.append({
            "question": q["question"],
            "role": q["role"],
            "canonical_answer": q.get("canonical_answer", ""),
            "distance": float(distances[0][i])
        })
        if len(results) >= top_k:
            break
    return results


# ========== 5️⃣ Demo ==========
if __name__ == "__main__":
    questions = load_questions()
    index, _ = build_index(questions)

    # Example test query
    role = "Product Manager"
    resume = "Led a team developing ML pipelines in Python, handled client communications and project delivery."
    job_desc = "Looking for a product manager experienced in leadership, agile workflows, and data analysis."

    test_query = "leadership and project management experience"
    results = search(test_query, questions, index, role=role, resume=resume, job_desc=job_desc)

    print("\n🔍 Retrieved questions:")
    for r in results:
        print(f"- {r['question']} (role: {r['role']}, score: {r['distance']:.2f})")
