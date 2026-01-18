ADVANCED MATHEMATICS SOLVER AGENT

An AI-powered advanced mathematics problem solver with multi-modal input support (text, image, audio) and human-in-the-loop verification. This application is specifically designed for JEE (Joint Entrance Examination) Advanced mathematics problems, providing step-by-step solutions with rigorous mathematical reasoning.

FEATURES

* Multi-Model AI Solver
  - Supports Groq, OpenAI, and Gemini models
  - Automatic fallback between models for reliability
  - Rigorous solution verification

* Input Methods
  - Text input for direct problem entry
  - Image upload with OCR (Optical Character Recognition)
  - Audio recording/upload with speech-to-text

* Advanced Features
  - Human-in-the-loop verification system
  - Persistent memory for learned solutions
  - Solution verification and validation
  - Detailed step-by-step solutions

QUICK START

PREREQUISITES:
- Python 3.9+
- Tesseract OCR (for image processing)
- FFmpeg (for audio processing)
- MongoDB (for solution storage)
- API keys for Groq, OpenAI, and Gemini

INSTALLATION:
1. Clone the repository:
   git clone https://github.com/yourusername/math-solver-agent.git
   cd math-solver-agent/backend

2. Set up a virtual environment:
   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux:
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Environment Variables:
   Create a .env file with the following variables:
   
   # Required API Keys
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MONGODB_URI=mongodb://localhost:27017/jee_solver
   
   # Optional Model Configuration
   GROQ_MODEL=mixtral-8x7b-32768
   OPENAI_MODEL=gpt-4
   GEMINI_MODEL=gemini-pro
   
   # Feature Toggles
   ENABLE_VERIFICATION=true
   ENABLE_MEMORY=true
   
   # OCR Configuration
   TESSERACT_CMD=/usr/bin/tesseract
   
   # Audio Configuration
   AUDIO_SAMPLE_RATE=16000

RUNNING THE APPLICATION:
1. Start the Streamlit server:
   streamlit run app.py

2. Access the web interface at http://localhost:8501

3. Input your problem using any of these methods:
   - Type directly in the text area
   - Upload an image containing the problem
   - Record or upload an audio file

4. Configure settings (optional):
   - Select AI model (Auto, Groq, OpenAI, or Gemini)
   - Enable/disable verification
   - Toggle memory usage

5. Click "Solve" to get a step-by-step solution

PROJECT ARCHITECTURE


CORE COMPONENTS:
1. app.py
   - Main Streamlit application
   - Handles user interface and input processing
   - Coordinates between different modules

2. imagetotext.py
   - Image preprocessing and enhancement
   - Multi-engine OCR (Tesseract + EasyOCR)
   - Text extraction from mathematical expressions

3. audiototext.py
   - Audio file handling
   - Speech-to-text conversion
   - Multiple fallback mechanisms

4. solver.py
   - Core mathematical problem-solving logic
   - Model selection and fallback mechanisms
   - Solution formatting and validation

5. rag.py
   - Retrieval-Augmented Generation implementation
   - Context management for problem-solving
   - Knowledge base integration

6. memory.py
   - Persistent storage of solutions
   - Retrieval of similar past solutions
   - MongoDB integration

7. ocr_verify.py
   - OCR result validation
   - Confidence scoring
   - Error correction

API DOCUMENTATION

extract_text_from_image(image_file: bytes) -> str
Extracts text from an image using OCR.

Parameters:
- image_file: Binary image data

Returns:
- Extracted text as a string

extract_text_from_audio(audio_bytes: bytes) -> str
Converts speech to text from audio data.

Parameters:
- audio_bytes: Binary audio data

Returns:
- Transcribed text as a string

solve_with_groq_rag(question: str) -> str
Solves a math problem using Groq's LLM with RAG.

Parameters:
- question: The math problem to solve

Returns:
- Formatted solution as a string

DEVELOPMENT

ADDING NEW FEATURES:
1. Create a new branch:
   git checkout -b feature/new-feature

2. Make your changes and test thoroughly

3. Run tests:
   python -m pytest tests/

4. Submit a pull request

TESTING:
Run the test suite:
pytest tests/

LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.

ACKNOWLEDGMENTS

- Tesseract OCR for text recognition
- OpenAI for speech-to-text capabilities
- Groq and Google for their powerful language models
- Streamlit for the web interface

CONTACT
For questions or support, please open an issue on GitHub or contact the maintainers.