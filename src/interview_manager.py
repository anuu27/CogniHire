import json
from retrieval import load_index, search
from generator import generate_followup
from resume_utils import get_resume_context   # 🆕 import improved resume parser
from datetime import datetime

# ---------- 1️⃣ Resume-based round ----------
def resume_based_round(questions, index, role, resume_text, job_desc, log):
    print("\n========== 🧾 RESUME-BASED QUESTIONS ==========")

    # 🧠 Extract structured context
    context = get_resume_context(resume_text)
    topics = context["keywords"] + context["phrases"]  # combine top keywords + phrases
    summary = context["summary"]

    # Limit to 4 main resume topics
    for topic in topics[:4]:
        print(f"\n🎯 Topic: {topic}")
        results = search(
            topic,
            questions,
            index,
            role=role,
            resume=summary,  # 🧩 pass summary not full text
            job_desc=job_desc,
            top_k=1
        )

        if results:
            q = results[0]["question"]
            canonical = results[0].get(
                "canonical_answer",
                "Expected detailed discussion of skill and outcomes."
            )
        else:
            q = f"Can you tell me more about your experience with {topic}?"
            canonical = "Expected detailed discussion of experience and outcomes."

        print("❓", q)
        ans = input("👤 Your answer: ").strip()
        follow = generate_followup(ans, canonical, q)
        print("🤖 Follow-up:", follow)

        log.append({
            "round": "resume",
            "topic": topic,
            "question": q,
            "answer": ans,
            "follow_up": follow
        })


# ---------- 2️⃣ Role-based round ----------
def role_based_round(questions, index, role, log):
    print("\n========== 💼 ROLE-BASED QUESTIONS ==========")
    results = search(f"basic questions for {role}", questions, index, role=role, top_k=5)
    for i, r in enumerate(results, 1):
        q = r["question"]
        print(f"\nQ{i}: {q}")
        ans = input("👤 Your answer: ").strip()
        log.append({"round": "role", "question": q, "answer": ans})


# ---------- 3️⃣ Interview controller ----------
def run_interview(role, resume_text, job_desc):
    print(f"\n🎙️ Starting virtual interview for: {role}")
    questions, index = load_index()
    transcript = []

    # Two stages: Resume-based + Role-based
    resume_based_round(questions, index, role, resume_text, job_desc, transcript)
    role_based_round(questions, index, role, transcript)

    # Save transcript
    out_path = f"data/interview_log_{role.replace(' ','_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "interview": transcript
        }, f, indent=2)

    print(f"\n✅ Interview complete. Transcript saved to {out_path}")


# ---------- 4️⃣ Example manual run ----------
if __name__ == "__main__":
    role = "Product Manager"
    resume_text = """
    Experienced Product Manager with 5 years leading cross-functional teams to design
    and launch SaaS products. Skilled in stakeholder management, UX research, data analytics,
    and agile methodologies. Proficient in Python and SQL for data-driven decisions.
    """
    job_desc = "Looking for a PM skilled in leadership, communication, agile delivery, and analytics."

    run_interview(role, resume_text, job_desc)
