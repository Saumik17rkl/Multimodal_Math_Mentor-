import streamlit as st
import os
from groq import Groq
from google import genai
from pymongo import MongoClient
from google.genai.errors import ClientError
from hitl import init_hitl_fields, save_hitl_review
from openai import OpenAI
from memory import retrieve_memory, store_memory
from datetime import datetime
from rag import build_rag_prompt
from imagetotext import extract_text_from_image
from audiototext import extract_text_from_audio
from ocr_verfiy import verify_ocr_with_vision

from dotenv import load_dotenv
load_dotenv()


# ---------------- ENV VALIDATION ----------------
REQUIRED_VARS = [
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "MONGODB_URI",
    "GROQ_MODEL",
    "GEMINI_MODEL",
    "OPENAI_MODEL"
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]

if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

# ---------------- CLIENTS ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["jee_solver"]
collection = db["solutions"]

GROQ_MODEL = os.getenv("GROQ_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

SYSTEM_PROMPT = """
You are an expert JEE (Joint Entrance Examination) Advanced mathematics problem solver with deep knowledge of calculus, algebra, coordinate geometry, trigonometry, vectors, and probability.

CORE PRINCIPLES:
1. Mathematical rigor is paramount - every step must be logically sound
2. Clarity over brevity - explain reasoning at each stage
3. Verify your work - check dimensional consistency and edge cases
4. Be honest about limitations - state when a problem requires numerical methods or is beyond scope

SOLUTION STRUCTURE:
1. **Problem Analysis**
   - Identify the topic area (calculus, algebra, geometry, etc.)
   - List given information and what needs to be found
   - Note any constraints or special conditions

2. **Approach Selection**
   - State your chosen method and why it's appropriate
   - Mention alternative approaches if applicable
   - Highlight any key theorems or formulas you'll use

3. **Step-by-Step Solution**
   - Number each step clearly
   - Show ALL algebraic manipulations (don't skip steps)
   - Explain the reasoning behind non-obvious transitions
   - Use proper mathematical notation (∫, ∑, lim, etc.)
   - For graphs/diagrams, describe them clearly in words

4. **Verification**
   - Check if the answer satisfies original constraints
   - Verify units/dimensions if applicable
   - Test with limiting cases or special values when possible
   - Cross-check using alternative methods if feasible

5. **Final Answer**
   - Present the answer in a boxed format: ■ Answer: [result]
   - Include units if applicable
   - State the answer in simplest form (rationalized, factored, etc.)

GUARDRAILS:
✓ DO:
- Use standard mathematical notation and terminology
- Explain WHY a formula or theorem applies, not just THAT it does
- Break complex problems into manageable sub-problems
- Point out common pitfalls or mistakes students make
- Provide geometric or physical intuition when helpful
- State domain restrictions and conditions of validity

✗ DON'T:
- Skip algebraic steps or say "it's obvious that..."
- Use calculator approximations unless specifically requested
- Ignore edge cases or domain restrictions
- Confuse sufficient and necessary conditions
- Make unjustified assumptions about problem constraints
- Present incorrect work confidently

UNCERTAINTY HANDLING:
If you encounter:
- Ambiguous problem statements → Ask for clarification
- Multiple valid interpretations → Present all possibilities
- Computational complexity beyond symbolic methods → State this explicitly
- Gaps in your reasoning → Mark with "Note: This step requires verification"
- Errors in your work → Acknowledge and correct immediately

COMMON JEE TOPICS CHECKLIST:
- Calculus: Limits, continuity, derivatives, integrals, differential equations
- Algebra: Complex numbers, quadratics, sequences/series, inequalities, polynomials
- Coordinate Geometry: Straight lines, circles, parabola, ellipse, hyperbola
- Trigonometry: Identities, equations, inverse functions, properties of triangles
- Vectors & 3D: Dot/cross products, lines, planes, spheres
- Probability: Conditional probability, distributions, combinatorics
- Matrices & Determinants: Operations, eigenvalues, system of equations

FORMATTING:
- Use LaTeX-style notation where helpful: \frac{a}{b}, \sqrt{x}, x^2, \int, \sum
- For fractions, use proper notation: 3/4 or \frac{3}{4}
- Clearly distinguish between multiplication (·, ×, or juxtaposition)
- Use parentheses liberally to avoid ambiguity

IMPORTANT OUTPUT FORMAT (STRICT):

Solve the question exactly like a JEE answer written on paper.

DO:
- Write only mathematical steps.
- Use direct algebraic manipulation.
- Start with "Given:" if values are provided.
- Show each step using "=" or "⇒" on a new line.
- Keep explanations minimal and inline (if any).
- End with "∴ Answer: ..."

AFTER the solution, add ONLY ONE short section:

Concept Used:
[Name of the concept in one line only]

DO NOT:
- Do NOT write STEP 1, STEP 2, etc.
- Do NOT write analysis, verification, or reasonableness.
- Do NOT mention reference material or RAG.
- Do NOT explain theory in paragraphs.
- Do NOT add units check unless explicitly required.
- Do NOT use headings other than "Given" and "Concept Used".

Any output that includes analysis sections, verification text, or meta commentary is INVALID.


"""

# ---------------- LLM FUNCTIONS ----------------
def solve_with_groq_rag(question: str):
    try:
        prompt = build_rag_prompt(question)

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.15
        )
        return response.choices[0].message.content
    except Exception:
        return None

def solve_with_groq_rag_safe(question: str):
    # First attempt (strict)
    answer = solve_with_groq_rag(question)

    # If empty or too short, retry with relaxed temperature
    if not answer or len(answer.strip()) < 50:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_rag_prompt(question)}
            ],
            temperature=0.35  # relaxed
        )
        answer = response.choices[0].message.content

    return answer

def solve_with_openai_rag(question: str):
    try:
        prompt = build_rag_prompt(question)
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception:
        return None

def solve_with_gemini_rag(question: str):
    try:
        prompt = build_rag_prompt(question)
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=SYSTEM_PROMPT + "\n\n" + prompt
        )
        return response.text

    except ClientError as e:
        # Gemini quota / rate-limit / billing errors
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return None
        raise e  # unknown Gemini error → surface it


def is_acceptable_answer(answer: str) -> bool:
    if not answer:
        return False

    # Explicit refusal only
    refusal_signals = [
        "insufficient information",
        "cannot solve",
        "unable to solve",
        "cannot determine",
        "outside scope"
    ]

    text = answer.lower()
    return not any(s in text for s in refusal_signals)


def save_to_db(question, solution, model_used):
    collection.insert_one({
        "question": question,
        "solution": solution,
        "model_used": model_used,
        "timestamp": datetime.utcnow(),
        **init_hitl_fields()
    })

def is_real_failure(answer: str) -> bool:
    if answer is None:
        return True

    hard_fail_signals = [
        "i cannot solve",
        "i am unable",
        "cannot be determined",
        "no solution exists",
        "outside my scope",
        "unable to proceed",
        "cannot answer"
    ]

    text = answer.lower()
    return any(signal in text for signal in hard_fail_signals)

# ================= STREAMLIT UI =================
st.set_page_config("JEE Math Solver Agent", layout="centered")
st.title("📐 JEE Math Solver Agent")

# ================= INPUT MODE =================
input_mode = st.radio(
    "Choose input type",
    ["Text", "Image", "Audio","PDF"],
    horizontal=True
)

question = ""

# ---------- TEXT ----------
if input_mode == "Text":
    question = st.text_area(
        "Enter a Math Question (JEE Level)",
        height=150
    )

# ---------- IMAGE ----------
elif input_mode == "Image":
    uploaded_image = st.file_uploader("Upload question image", ["png", "jpg", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image)

        with st.spinner("Running OCR (multiple engines)..."):
            extracted = extract_text_from_image(uploaded_image)

        question = st.text_area(
            "Verify / correct extracted text",
            value=extracted,
            height=150
        )

# ---------- AUDIO ----------
elif input_mode == "Audio":
    recorded_audio = st.audio_input("Record your question")

    if recorded_audio:
        with st.spinner("Transcribing audio..."):
            extracted = extract_text_from_audio(recorded_audio.getvalue())

        question = st.text_area(
            "Verify / correct transcribed text",
            value=extracted,
            height=150
        )

elif input_mode == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        extracted = extract_text_from_pdf(uploaded_pdf)

        question_text = st.text_area(
            "Verify / correct extracted text",
            value=extracted,
            height=200
        )


if "solution" not in st.session_state:
    st.session_state.solution = None

if "question" not in st.session_state:
    st.session_state.question = None


# ================= SOLVE =================
if st.button("Solve"):
    if not question.strip():
        st.warning("Enter a valid question.")
        st.stop()

    solution = None
    model_used = None

    # ---------- MEMORY FIRST ----------
    memory_hit = retrieve_memory(question)

    if memory_hit:
        st.info("📚 Answer retrieved from learned memory")
        solution = memory_hit["solution"]
        model_used = "memory"

    else:
        with st.spinner("Solving using Groq + RAG..."):
            solution = solve_with_groq_rag_safe(question)
            model_used = "groq+rag"

        if not is_acceptable_answer(solution):
            st.warning("Groq insufficient. Falling back to Gemini + RAG...")
            with st.spinner("Solving using Gemini + RAG..."):
                solution = solve_with_gemini_rag(question)
                model_used = "gemini+rag"

        if not is_acceptable_answer(solution):
            st.warning("Gemini insufficient. Falling back to OpenAI + RAG...")
            with st.spinner("Solving using OpenAI + RAG..."):
                solution = solve_with_openai_rag(question)
                model_used = "openai+rag"

    if not solution:
        st.error("All models failed. Question may be ambiguous or out of scope.")
        st.stop()

    # ✅ STORE IN SESSION STATE (ONLY HERE)
    st.session_state.solution = solution
    st.session_state.question = question

    save_to_db(question, solution, model_used)

    st.markdown("### ✅ Solution")
    st.markdown(st.session_state.solution)

    # ================= HITL (Human-In-The-Loop) =================
    if st.session_state.solution is not None:
        st.markdown("## 🧑‍⚖️ Human Review (HITL)")

        decision = st.radio(
            "Review decision",
            ["Not reviewed", "Approve", "Reject", "Edit & Approve"],
            index=0,
            key=f"hitl_decision_{st.session_state.question}"
        )

        feedback = st.text_area(
            "Reviewer feedback (optional)",
            key=f"hitl_feedback_{st.session_state.question}"
        )

        corrected_solution = None
        if decision == "Edit & Approve":
            corrected_solution = st.text_area(
                "Corrected Solution",
                value=st.session_state.solution,
                height=300,
                key=f"hitl_corrected_{st.session_state.question}"
            )

        try:
            if decision in ["Approve", "Edit & Approve"]:
                final_solution = corrected_solution if decision == "Edit & Approve" else st.session_state.solution
                
                # Store in memory for future use
                store_memory(
                    question=st.session_state.question,
                    solution=final_solution,
                    concept="JEE Mathematics",
                    approved=True,
                    source="human"
                )

                # Save HITL review to database
                save_hitl_review(
                    collection=collection,
                    question=st.session_state.question,
                    original_solution=st.session_state.solution,
                    decision=decision,
                    feedback=feedback,
                    corrected_solution=corrected_solution
                )
                st.success("✔ Human-approved solution learned permanently")

            elif decision == "Reject":
                save_hitl_review(
                    collection=collection,
                    question=st.session_state.question,
                    original_solution=st.session_state.solution,
                    decision=decision,
                    feedback=feedback,
                    corrected_solution=None
                )
                st.warning("❌ Solution rejected (not learned)")
                
        except Exception as e:
            st.error(f"❌ Error processing review: {str(e)}")
            st.exception(e)  # Show detailed error in debug mode

# ================= HISTORY (READ-ONLY) =================
with st.expander("📜 Previous Questions"):
    for item in collection.find().sort("timestamp", -1).limit(5):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**Solved by:** `{item['model_used']}`")
        st.markdown(item["solution"])
        st.markdown(f"**Review status:** `{item.get('hitl_decision', 'Not reviewed')}`")
        st.markdown("---")
