from memory import retrieve_memory, store_memory
from rag import build_rag_prompt
from llms import (
    solve_with_groq,
    solve_with_gemini,
    solve_with_openai
)
from utils import is_acceptable_answer

def solve_question(question: str):
    # 1️⃣ Memory
    memory_hit = retrieve_memory(question)
    if memory_hit:
        return {
            "source": "memory",
            "solution": memory_hit["solution"]
        }

    # 2️⃣ Groq
    solution = solve_with_groq(question)
    source = "groq"

    # 3️⃣ Gemini
    if not is_acceptable_answer(solution):
        solution = solve_with_gemini(question)
        source = "gemini"

    # 4️⃣ OpenAI
    if not is_acceptable_answer(solution):
        solution = solve_with_openai(question)
        source = "openai"

    if not solution:
        return {
            "error": "All models failed"
        }

    return {
        "source": source,
        "solution": solution
    }
