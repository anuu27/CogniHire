from transformers import pipeline
import json

def generate_questions_for_role(role, num=5):
    print(f"Generating {num} questions for {role}")
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",  # ✅ even lighter
        device="cpu"
    )
    prompt = f"Generate {num} interview questions for a {role} role."
    out = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    return out

if __name__ == "__main__":
    roles = ["Product Manager", "Data Analyst"]
    all_questions = {}
    for r in roles:
        all_questions[r] = generate_questions_for_role(r, 5)
    with open("data/generated_questions.json", "w") as f:
        json.dump(all_questions, f, indent=2)
    print("✅ Saved generated questions to data/generated_questions.json")
