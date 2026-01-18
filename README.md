# Advanced mathematics solver agent

An AI-powered advanced mathematics problem solver with multi-modal input support (text, image, audio, PDF) and human-in-the-loop verification.

## Features

- 🧠 Multiple AI model support (Groq, OpenAI, Gemini)
- 📸 Image-to-text conversion with OCR
- 🎤 Speech-to-text for audio input
- 📄 PDF text extraction
- 👥 Human-in-the-loop verification
- 💾 Persistent memory for learned solutions

## Prerequisites

- Python 3.9+
- Tesseract OCR (for local development)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/math-solver-agent.git
   cd math-solver-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   # API Keys
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MONGODB_URI=your_mongodb_uri
   
   # Model Names (optional, defaults will be used if not set)
   GROQ_MODEL=mixtral-8x7b-32768
   OPENAI_MODEL=gpt-4
   GEMINI_MODEL=gemini-pro
   ```

## Running Locally

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser to `http://localhost:8501`

## Deployment

### Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your forked repository
4. Set the following secrets in Streamlit Cloud:
   - `GROQ_API_KEY`
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `MONGODB_URI`
5. Set the main file path to `streamlit_app.py`
6. Deploy!

## Project Structure

```
.
├── backend/
│   ├── app.py              # Main Streamlit application
│   ├── requirements.txt     # Python dependencies
│   ├── rag.py              # Retrieval-Augmented Generation logic
│   ├── memory.py           # Memory management
│   ├── hitl.py             # Human-in-the-loop functionality
│   ├── imagetotext.py      # Image processing and OCR
│   ├── audiototext.py      # Audio processing and transcription
│   └── ocr_verify.py       # OCR verification with AI
├── streamlit_app.py        # Streamlit entry point
└── .gitignore
```

## License

MIT License - see the [LICENSE](LICENSE) file for details.
