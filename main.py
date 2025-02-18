import uvicorn
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
import torch
import torch.quantization
import nest_asyncio

# Initialize FastAPI app
app = FastAPI()

# Global model cache
model_cache = {"model": None, "tokenizer": None, "device": None}

# Function to lazily load and quantize the model
def get_model():
    if model_cache["model"] is None:
        print("Loading and quantizing model for the first time...")

        model_name = "facebook/m2m100_418M"
        model_cache["tokenizer"] = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        # Set device
        device = torch.device("cpu")  # CPU for AWS Free Tier
        model.to(device)
        model.eval()  # Set to evaluation mode

        # 🔹 Apply dynamic quantization (only works on CPU)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8  # Reduces memory usage
        )

        # Cache the model and tokenizer
        model_cache["model"] = model
        model_cache["device"] = device

        print("Model successfully loaded and quantized!")

    return model_cache["model"], model_cache["tokenizer"], model_cache["device"]

# Supported languages (static)
supported_languages = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M").lang_code_to_id.keys()

# Request model for translation
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Translation endpoint
@app.post("/translate/")
async def translate(request: TranslationRequest, model_data=Depends(get_model)):
    try:
        model, tokenizer, device = model_data

        # Tokenize the input text
        tokenizer.src_lang = request.source_lang
        encoded_text = tokenizer(request.text, return_tensors="pt").to(device)

        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_text,
                forced_bos_token_id=tokenizer.get_lang_id(request.target_lang),
            )

        # Decode the translation
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return {"translated_text": translated_text}

    except Exception as e:
        return {"error": str(e)}

# Language detection endpoint
@app.get("/detect_language/")
async def detect_language(text: str):
    try:
        detected_lang = detect(text)
        return {"detected_language": detected_lang}
    except Exception as e:
        return {"error": str(e)}

# Supported languages endpoint
@app.get("/supported_languages/")
async def get_supported_languages():
    return {"languages": list(supported_languages)}

# Main entry point for development
if __name__ == "__main__":
    
    config = uvicorn.Config(app)
    server = uvicorn.Server(config)
    server.run()
