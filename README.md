

# 🧠 Multimodal Math Mentor – Multi-Agent AI Tutor (FastAPI)

An **AI-powered multi-agent mathematics tutor** designed for **JEE Advanced–level problem solving**, supporting **text, image, and audio inputs** with **rigorous step-by-step reasoning**, **verification**, and **human-in-the-loop (HITL)** validation.

This project is built as a **FastAPI backend service**, intended to be consumed by any frontend (Web, Mobile, Streamlit, React, etc.).

---

## 🚀 Key Capabilities

### 🧩 Multi-Agent Architecture

The system is not a single LLM call. It uses **specialized agents**, each with a defined role:

* **OCR Agent** – Extracts math text from images
* **Speech Agent** – Converts audio → text
* **Solver Agent** – Performs structured mathematical reasoning
* **Verifier Agent** – Validates correctness of each step
* **RAG Agent** – Injects retrieved context when needed
* **Memory Agent** – Stores & recalls solved problems
* **HITL Agent** – Enables human validation when confidence is low

---

### 🧠 Multi-Model AI Support

* Groq
* OpenAI
* Gemini
  With **automatic fallback** for reliability.

---

### 📥 Multimodal Input

* ✅ Text
* 🖼️ Image (OCR + verification)
* 🎙️ Audio (speech-to-text)

---

## 🏗️ Architecture Overview

**Backend-first design (NO Streamlit dependency).**

```
Client (Web / App / Streamlit)
        ↓
     FastAPI
        ↓
 ┌──────────────────────────┐
 │  Multi-Agent AI Pipeline │
 └──────────────────────────┘
```

---

## 📂 Project Structure (UPDATED)

```
.
├── main.py              # 🚀 FastAPI application entry point
├── wsgi.py              # Deployment entry (Gunicorn/Uvicorn)
├── app.py               # Optional UI / testing layer (NOT main server)
├── solver.py            # Core math reasoning agent
├── rag.py               # Retrieval-Augmented Generation agent
├── memory.py            # Persistent memory (MongoDB)
├── hitl.py              # Human-in-the-loop verification
├── imagetotext.py       # Image → text (OCR agent)
├── audiototext.py       # Audio → text (Speech agent)
├── ocr_verify.py        # OCR confidence & correction agent
├── lldhld.py            # Low-level decision logic
├── requirements.txt
├── Procfile             # Deployment config
├── .env.example
├── README.md
└── .gitignore
```

👉 **Important:**
`main.py` is the **only backend server**.
`app.py` is **not** the production entry point.

---

## ⚙️ Prerequisites

* Python **3.9+**
* Tesseract OCR
* FFmpeg
* MongoDB
* API keys:

  * Groq
  * OpenAI
  * Gemini

---

## 🛠️ Installation

```bash
git clone https://github.com/Saumik17rkl/Multimodal_Math_Mentor-.git
cd Multimodal_Math_Mentor-
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_MODEL=gemini-pro

ENABLE_VERIFICATION=true
ENABLE_MEMORY=true

TESSERACT_CMD=/usr/bin/tesseract
AUDIO_SAMPLE_RATE=16000
```

---

## ▶️ Running the Backend (CORRECT WAY)

### 🔥 Start FastAPI Server

```bash
uvicorn main:app --reload
```

Server runs at:

```
http://localhost:8000
```

API Docs:

```
http://localhost:8000/docs
```

---

## 🔌 Core API Capabilities

### Image → Math Problem

```python
extract_text_from_image(image_bytes) -> str
```

### Audio → Math Problem

```python
extract_text_from_audio(audio_bytes) -> str
```

### Solve with Multi-Agent RAG

```python
solve_with_groq_rag(question: str) -> str
```

Each request flows through:
**OCR → Verification → Solver → Verifier → Memory → Response**

---

## 🧪 Development & Testing

```bash
pytest tests/
```

---

## 🧠 Why This Project Is Different

❌ Not a demo chatbot
❌ Not a single-prompt LLM wrapper
❌ Not Streamlit-dependent

✅ True **multi-agent reasoning system**
✅ Backend-first, scalable architecture
✅ Designed for **serious competitive mathematics**

---

## 📜 License

MIT License

---

## 📬 Contact

Open an issue on GitHub for discussions or improvements.
