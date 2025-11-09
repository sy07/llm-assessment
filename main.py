import os
import json
import re
import openai

"""
With two more hours, I would add a small config layer and CLI flags for target age, word count, and rounds.
I would persist stories and judge reports to JSON files, introduce unit tests for the JSON schema and retry logic, and add basic telemetry for token usage and latency.
I would also create a handful of seed prompts for reproducible evaluation, implement a lightweight safety checklist before judging, and include a short README section on results, costs, and limitations.
"""

MODEL = "gpt-3.5-turbo" 

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Low-level chat helper that supports system + user messages ---
def chat(messages, temperature=0.2, max_tokens=1200):
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return resp.choices[0].message["content"]

# --- Prompts ---
STORYTELLER_SYSTEM = """You are a kind children's storyteller writing for ages 5–10.
Write warm, imaginative stories with:
- A clear beginning, middle, and end (gentle conflict & resolution)
- Friendly characters and vivid but simple imagery
- A soft moral (kindness, curiosity, perseverance) without preaching
- Simple sentences and age-appropriate vocabulary
- Safe, comforting tone (avoid violence, fear, complex romance, or dark themes)
Aim for ~400–700 words unless the user requests otherwise.
Keep paragraphs short (2–4 sentences)."""

JUDGE_SYSTEM = """You are a meticulous children's story judge for ages 5–10.
Evaluate the story against this rubric:
1) Age fit & safety
2) Story arc clarity (beginning–middle–end, conflict & resolution)
3) Characters & setting (friendly, vivid, imaginative)
4) Moral/theme (gentle, uplifting, integrated)
5) Clarity & length (simple sentences; target ~400–700 words unless specified)
5) Engagement (rhythm, sensory details, gentle humor)
Return STRICT JSON with keys:
- overall_score: number 0–10
- scores: object with keys ["age_fit","arc","characters","moral","clarity_length","engagement"], each 0–10
- suggestions: array of concrete, concise improvements
- blocking_issues: array of any safety/age-appropriateness issues (may be empty)
- length_words: integer word count
Do NOT include any prose or code fences outside JSON. Only output JSON."""

REVISION_SYSTEM = """You are revising a children's story for ages 5–10.
Apply ONLY the requested improvements from the judge feedback while preserving what already works.
Keep the tone safe, warm, and comforting. Keep ~400–700 words unless specified.
Return ONLY the revised story text; no commentary, no headings."""

# --- Functions for the pipeline ---
def generate_initial_story(user_request: str) -> str:
    messages = [
        {"role": "system", "content": STORYTELLER_SYSTEM},
        {"role": "user", "content": f"Please write a bedtime story. User request/theme: {user_request}"},
    ]
    return chat(messages, temperature=0.5, max_tokens=1200)

def judge_story(story: str) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Here is the story to evaluate:\n\n{story}"},
    ]
    raw = chat(messages, temperature=0.0, max_tokens=700)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {raw[:300]}")
    data = json.loads(match.group(0))
    assert "overall_score" in data and "suggestions" in data and "scores" in data
    return data

def revise_story(story: str, judge_json: dict, user_request: str) -> str:
    feedback = json.dumps(
        {
            "overall_score": judge_json.get("overall_score"),
            "suggestions": judge_json.get("suggestions", []),
            "blocking_issues": judge_json.get("blocking_issues", []),
            "keep_length_target_words": 550,
        }
    )
    messages = [
        {"role": "system", "content": REVISION_SYSTEM},
        {"role": "user", "content": (
            "Revise the following story for ages 5–10 based on the judge feedback JSON.\n"
            f"User request/theme: {user_request}\n\n"
            f"JUDGE_JSON:\n{feedback}\n\n"
            f"STORY:\n{story}"
        )},
    ]
    return chat(messages, temperature=0.4, max_tokens=1200)

def generate_with_judge(user_request: str, min_score: float = 8.5, max_rounds: int = 3):
    story = generate_initial_story(user_request)
    for _ in range(1, max_rounds + 1):
        result = judge_story(story)
        score = float(result.get("overall_score", 0))
        if score >= min_score and not result.get("blocking_issues"):
            return story, result, _
        story = revise_story(story, result, user_request)
    final_result = judge_story(story)
    return story, final_result, max_rounds

def main():
    user_input = input("What kind of story do you want to hear? ")
    story, judge, rounds = generate_with_judge(user_input)
    print("\n" + "#"*80 + "\nFINAL STORY\n" + "#"*80 + "\n")
    print(story)
    print("\n" + "#"*80 + "\nJUDGE REPORT (JSON)\n" + "#"*80 + "\n")
    print(json.dumps(judge, indent=2))

if __name__ == "__main__":
    main()
