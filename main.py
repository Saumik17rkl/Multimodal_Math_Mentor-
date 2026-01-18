from app import SYSTEM_PROMPT
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import os

from groq import Groq
from google import genai
from openai import OpenAI
from pymongo import MongoClient
from google.genai.errors import ClientError

from rag import build_rag_prompt
from imagetotext import extract_text_from_image
from audiototext import extract_text_from_audio
from memory import retrieve_memory, store_memory
from hitl import init_hitl_fields, save_hitl_review

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
    raise RuntimeError(f"Missing env vars: {missing}")
SYSTEM_PROMPT = """You are an expert JEE (Joint Entrance Examination) Advanced mathematics problem solver with deep knowledge of calculus, algebra, coordinate geometry, trigonometry, vectors, and probability.

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

# ---------------- CLIENTS ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_MODEL = os.getenv("GROQ_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

mongo = MongoClient(os.getenv("MONGODB_URI"))
db = mongo["jee_solver"]
solutions_col = db["solutions"]

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="JEE Math Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LLM FUNCTIONS ----------------
def solve_with_groq_rag_safe(question: str):
    try:
        prompt = build_rag_prompt(question)
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.15
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def solve_with_gemini_rag(question: str):
    try:
        prompt = build_rag_prompt(question)
        resp = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=SYSTEM_PROMPT + "\n\n" + prompt
        )
        return resp.text
    except ClientError:
        return None


def solve_with_openai_rag(question: str):
    try:
        prompt = build_rag_prompt(question)
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def is_acceptable_answer(ans: Optional[str]) -> bool:
    if not ans:
        return False
    bad = [
        "cannot solve",
        "unable to solve",
        "outside scope",
        "cannot determine"
    ]
    text = ans.lower()
    return not any(b in text for b in bad)


# ---------------- CORE SOLVE PIPELINE ----------------
def solve_question(question: str):
    memory_hit = retrieve_memory(question)
    if memory_hit:
        return memory_hit["solution"], "memory"

    sol = solve_with_groq_rag_safe(question)
    model = "groq+rag"

    if not is_acceptable_answer(sol):
        sol = solve_with_gemini_rag(question)
        model = "gemini+rag"

    if not is_acceptable_answer(sol):
        sol = solve_with_openai_rag(question)
        model = "openai+rag"

    if not sol:
        raise HTTPException(400, "All models failed")

    solutions_col.insert_one({
        "question": question,
        "solution": sol,
        "model_used": model,
        "timestamp": datetime.utcnow(),
        **init_hitl_fields()
    })

    return sol, model


# ---------------- API ENDPOINTS ----------------

@app.get("/")
def main():
    return{"The application is working"}

@app.post("/solve/text")
async def solve_text(question: str = Form(...)):
    solution, model = solve_question(question)
    return {"question": question, "solution": solution, "model": model}


@app.post("/solve/image")
async def solve_image(file: UploadFile = File(...)):
    text = extract_text_from_image(file.file)
    return {"extracted_text": text}


@app.post("/solve/audio")
async def solve_audio(file: UploadFile = File(...)):
    text = extract_text_from_audio(await file.read())
    return {"extracted_text": text}


@app.post("/solve/confirm")
async def solve_confirm(question: str = Form(...)):
    solution, model = solve_question(question)
    return {"solution": solution, "model": model}


@app.post("/hitl/review")
async def hitl_review(
    question: str = Form(...),
    solution: str = Form(...),
    decision: str = Form(...),
    feedback: str = Form(""),
    corrected_solution: Optional[str] = Form(None)
):
    save_hitl_review(
        collection=solutions_col,
        question=question,
        original_solution=solution,
        decision=decision,
        feedback=feedback,
        corrected_solution=corrected_solution
    )

    if decision in ["Approve", "Edit & Approve"]:
        final = corrected_solution if corrected_solution else solution
        store_memory(
            question=question,
            solution=final,
            concept="JEE Mathematics",
            approved=True,
            source="human"
        )

    return {"status": "saved"}
