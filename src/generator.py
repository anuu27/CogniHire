from transformers import pipeline

# Use a small, reasoning-friendly open model
# If you have GPU + Hugging Face login, you can switch to:
# model="mistralai/Mistral-7B-Instruct-v0.2"
generator = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens=100,
    do_sample=False,
)

def generate_followup(candidate_answer, canonical_answer, question, resume=None, job_desc=None):
    """
    Generates ONE follow-up question if something is missing.
    Adds resume and job description context to make follow-ups more targeted.
    """

    # Build contextual prompt
    context = ""
    if resume:
        context += f"Candidate Resume Summary: {resume}\n"
    if job_desc:
        context += f"Job Description: {job_desc}\n"

    prompt = f"""
You are an AI interviewer evaluating candidate responses.

Question: {question}
Expected Answer: {canonical_answer}
Candidate's Answer: {candidate_answer}

Your task:
- If the candidate's answer missed something important, ask ONE concise, natural follow-up question to explore that missing aspect.
- If the candidate already covered everything well, reply exactly: "No follow-up needed."

Output ONLY the follow-up question or "No follow-up needed" — nothing else.
"""
    response = generator(prompt, max_new_tokens=80,do_sample=False,return_full_text=False)
    output = response[0]["generated_text"].split("Follow-up:")[-1].strip()

    # Keep only the first clean sentence
    output = output.split("\n")[0].strip()

    # Safety filter — avoid echoing prompt
    if output.lower().startswith(("question:", "expected", "candidate")) or len(output.split()) < 3:
        output = "No follow-up needed."

    return output


# ================== Demo Run ==================
if __name__ == "__main__":
    q = "Can you tell me about your leadership style?"
    canonical = "Expected competency: Candidate should describe leadership style, mentoring, team conflict resolution."
    candidate = "I usually lead by example, but I haven’t really mentored anyone formally."
    resume = "Led a 5-person ML team, managing project delivery and resolving internal conflicts."
    job_desc = "Looking for a leader skilled in mentoring, conflict resolution, and agile project management."

    print("🎤 Question:", q)
    print("👤 Candidate Answer:", candidate)
    print("📌 Canonical Answer:", canonical)
    print("📄 Resume:", resume)
    followup = generate_followup(candidate, canonical, q, resume, job_desc)
    print("\n🤖 Follow-up Question:\n", followup)
