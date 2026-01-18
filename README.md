# Advanced Mathematics Solver Agent

An AI-powered advanced mathematics problem solver with multi-modal input support (text, image, audio) and human-in-the-loop verification. This application is specifically designed for JEE (Joint Entrance Examination) Advanced mathematics problems, providing step-by-step solutions with rigorous mathematical reasoning.

## 🌟 Features

- **Multi-Model AI Solver**
  - Supports Groq, OpenAI, and Gemini models
  - Automatic fallback between models for reliability
  - Rigorous solution verification

- **Input Methods**
  - Text input for direct problem entry
  - Image upload with OCR (Optical Character Recognition)
  - Audio recording/upload with speech-to-text

- **Advanced Features**
  - Human-in-the-loop verification system
  - Persistent memory for learned solutions
  - Solution verification and validation
  - Detailed step-by-step solutions

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Tesseract OCR (for image processing)
- FFmpeg (for audio processing)
- MongoDB (for solution storage)
- API keys for Groq, OpenAI, and Gemini

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/math-solver-agent.git
   cd math-solver-agent/backend
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file with the following variables:
   ```env
   # Required API Keys
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_MODEL=gemini-pro
   
   # Feature Toggles
   ENABLE_VERIFICATION=true
   ENABLE_MEMORY=true
   
   # OCR Configuration
   TESSERACT_CMD=/usr/bin/tesseract  # Path to Tesseract executable
   
   # Audio Configuration
   AUDIO_SAMPLE_RATE=16000
   ```

## 🚀 Running the Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface** at `http://localhost:8501`

3. **Input your problem** using any of these methods:
   - Type directly in the text area
   - Upload an image containing the problem
   - Record or upload an audio file

4. **Configure settings** (optional):
   - Select AI model (Auto, Groq, OpenAI, or Gemini)
   - Enable/disable verification
   - Toggle memory usage

5. **Click "Solve"** to get a step-by-step solution

## 🏗️ Project Architecture

### Core Components

1. **app.py**
   - Main Streamlit application
   - Handles user interface and input processing
   - Coordinates between different modules

2. **imagetotext.py**
   - Image preprocessing and enhancement
   - Multi-engine OCR (Tesseract + EasyOCR)
   - Text extraction from mathematical expressions

3. **audiototext.py**
   - Audio file handling
   - Speech-to-text conversion using OpenAI's Whisper
   - Temporary file management

4. **solver.py**
   - Core mathematical problem-solving logic
   - Model selection and fallback mechanisms
   - Solution formatting and validation

5. **rag.py**
   - Retrieval-Augmented Generation implementation
   - Context management for problem-solving
   - Knowledge base integration

6. **memory.py**
   - Persistent storage of solutions
   - Retrieval of similar past solutions
   - MongoDB integration

7. **ocr_verify.py**
   - OCR result validation
   - Confidence scoring
   - Error correction

## 🔍 API Documentation

### `extract_text_from_image(image_file: bytes) -> str`
Extracts text from an image using OCR.

**Parameters:**
- `image_file`: Binary image data

**Returns:**
- Extracted text as a string

### `extract_text_from_audio(audio_bytes: bytes) -> str`
Converts speech to text from audio data.

**Parameters:**
- `audio_bytes`: Binary audio data

**Returns:**
- Transcribed text as a string

### `solve_with_groq_rag(question: str) -> str`
Solves a math problem using Groq's LLM with RAG.

**Parameters:**
- `question`: The math problem to solve

**Returns:**
- Formatted solution as a string

## 🛠️ Development

### Adding New Features

1. Create a new branch:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Make your changes and test thoroughly

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

4. Submit a pull request

### Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Tesseract OCR for text recognition
- OpenAI for speech-to-text capabilities
- Groq and Google for their powerful language models
- Streamlit for the web interface

## 📬 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

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
