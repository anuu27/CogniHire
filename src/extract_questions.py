import pandas as pd
import json
import re
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "questions.json"

def find_csv_file():
    """Find the first CSV file in the data folder."""
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in data/. Please download dataset first.")
    return csv_files[0]

def is_valid_question(q):
    """Filter out non-question or personalized interviewer statements."""
    q_lower = q.lower()

    # Remove questions that are feedback-like or contain personal references
    bad_phrases = [
        "great example", "good example", "overall", "aditya", "you have", "i think",
        "your performance", "impressive", "as we discussed", "like you said",
        "feedback", "mentioned earlier", "good understanding"
    ]
    if any(bp in q_lower for bp in bad_phrases):
        return False

    # Question must end with a '?' and be reasonably long
    if not q.strip().endswith("?") or len(q.split()) < 5:
        return False

    # Avoid small talk questions
    if any(x in q_lower for x in ["how are you", "can you hear", "ready to begin"]):
        return False

    return True


def extract_interviewer_questions(transcript):
    """Extract interviewer questions from transcript text."""
    questions = []
    if not isinstance(transcript, str):
        return questions

    lines = re.split(r"[\n\r]", transcript)
    for line in lines:
        if line.lower().startswith("interviewer:"):
            q = line.split(":", 1)[1].strip()

            # 🧹 Clean feedback-like prefixes
            q = re.sub(
                r"^(that'?s\s+great[,!]*|that is\s+great[,!]*|good[,!]*|great[,!]*|excellent[,!]*|perfect[,!]*|okay[,!]*|alright[,!]*|wonderful[,!]*|thank you[,!]*|interesting[,!]*|nice[,!]*|cool[,!]*|awesome[,!]*|sure[,!]*|alright[,!]*|before we begin[,!]*|moving on[,!]*|now[,!]*)\s*",
                "",
                q,
                flags=re.I
            )

            # Remove mid-sentence filler like “that’s great, now”
            q = re.sub(r"\b(that'?s\s+great|excellent|good)\s*,?\s*(now|so|let'?s)\b", "", q, flags=re.I)

            # Clean up multiple spaces
            q = re.sub(r"\s+", " ", q).strip()

            if is_valid_question(q):
                questions.append(q)

    return questions



def build_question_bank(df):
    """Build structured question dataset from transcripts."""
    q_bank = []
    q_id = 0
    for _, row in df.iterrows():
        role = str(row.get("Role", "")).strip()
        decision = str(row.get("Decision", "")).strip()
        feedback = str(row.get("Reason_for_decision", "")).strip()
        job_desc = str(row.get("Job_Description", "")).strip()
        transcript = row.get("Transcript", "")

        questions = extract_interviewer_questions(transcript)
        for q in questions:
            q_id += 1

            # Canonical answer logic
            if feedback:
                canonical_answer = f"Expected competency: {feedback}"
            elif job_desc:
                canonical_answer = f"Relevant to role: {job_desc[:150]}..."
            else:
                canonical_answer = "General competency-based answer expected."

            q_bank.append({
                "id": f"q_{q_id:05d}",
                "role": role,
                "competency": "general",
                "difficulty": "medium",
                "question": q,
                "canonical_answer": canonical_answer,
                "rubric_points": [feedback] if feedback else [],
                "job_description": job_desc,
                "decision": decision
            })
    return q_bank


if __name__ == "__main__":
    csv_file = find_csv_file()
    print(f"📂 Using dataset: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} records from dataset")

    q_bank = build_question_bank(df)
    print(f"🧹 Filtering and saving cleaned questions...")

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(q_bank, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(q_bank)} clean questions → saved to {OUTPUT_FILE}")
