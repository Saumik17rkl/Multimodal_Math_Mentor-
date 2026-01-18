from PIL import Image, ImageEnhance
import pytesseract
import easyocr
import numpy as np
import cv2


# ---------- PREPROCESS ----------
def _preprocess(image: Image.Image) -> Image.Image:
    """
    Improve contrast + binarize for math OCR
    """
    img = image.convert("L")  # grayscale

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Convert to OpenCV
    img_np = np.array(img)

    # Adaptive threshold (better for scanned math)
    img_np = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return Image.fromarray(img_np)


# ---------- OCR 1: TESSERACT ----------
def ocr_tesseract(image: Image.Image) -> str:
    try:
        config = (
            "--psm 6 "
            "-c tessedit_char_whitelist="
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "+-=*/().,^√∫∑π"
        )
        return pytesseract.image_to_string(image, config=config)
    except Exception:
        return ""


# ---------- OCR 2: EASYOCR ----------
def ocr_easyocr(image: Image.Image) -> str:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        img_np = np.array(image)
        result = reader.readtext(img_np)
        return " ".join([text for (_, text, _) in result])
    except Exception:
        return ""


# ---------- MAIN API (UNCHANGED) ----------
def extract_text_from_image(image_file) -> str:
    """
    Multi-OCR with preprocessing + fallback.
    Function name and behavior preserved.
    """
    image = Image.open(image_file).convert("RGB")
    processed = _preprocess(image)

    # 1️⃣ Tesseract
    text = ocr_tesseract(processed)
    if len(text.strip()) > 20:
        return text.strip()

    # 2️⃣ EasyOCR
    text = ocr_easyocr(processed)
    if len(text.strip()) > 20:
        return text.strip()

    # 3️⃣ Raw Tesseract (no preprocessing fallback)
    text = ocr_tesseract(image)
    if len(text.strip()) > 20:
        return text.strip()

    # 4️⃣ Give whatever we got (user will edit)
    return text.strip()
