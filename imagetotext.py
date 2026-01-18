from PIL import Image
import pytesseract
import easyocr
import io

def ocr_tesseract(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""

def ocr_easyocr(image: Image.Image) -> str:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image)
        return " ".join([text for (_, text, _) in result])
    except Exception:
        return ""

def extract_text_from_image(image_file) -> str:
    image = Image.open(image_file).convert("RGB")

    # 1️⃣ Tesseract
    text = ocr_tesseract(image)
    if len(text.strip()) > 20:
        return text

    # 2️⃣ EasyOCR
    text = ocr_easyocr(image)
    if len(text.strip()) > 20:
        return text

    # 3️⃣ Last resort
    return text.strip()  # may be empty, user still edits
