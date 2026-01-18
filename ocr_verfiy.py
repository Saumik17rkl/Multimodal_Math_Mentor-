import os
from openai import OpenAI, RateLimitError, AuthenticationError

# Initialize client with API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def verify_ocr_with_vision(image_file, ocr_text):
    if client is None:
        return ocr_text  # Skip verification if no API key is set
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Correct OCR errors in this math question. Return only corrected text."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_file.getvalue().hex()}"
                            }
                        },
                        {"type": "text", "text": ocr_text}
                    ]
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except (RateLimitError, AuthenticationError) as e:
        # Handle rate limits and authentication errors
        print(f"OpenAI API error: {str(e)}")
        return ocr_text
    except Exception as e:
        print(f"Error in OCR verification: {str(e)}")
        return ocr_text
