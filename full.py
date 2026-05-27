import os
import base64
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image
import io
import json

load_dotenv()

# ─── Logger ───────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Config from .env ─────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")
VISION_MODEL   = os.getenv("VISION_MODEL")

# ─── OpenAI Client ────────────────────────────────────────────
client = AsyncOpenAI(api_key=OPENAI_API_KEY or "placeholder_key_not_set")

# ─── Output Folder ────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Router ───────────────────────────────────────────────────
router = APIRouter()

# ─── Constants ────────────────────────────────────────────────
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
VALID_SIZES   = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def resize_image_full(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Resize image keeping aspect ratio, convert to PNG."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[FULL RESIZE] {w}x{h} → {new_w}x{new_h}")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def image_to_base64_full(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_image_full(file: UploadFile, label: str):
    content_type = file.content_type or ""
    if not content_type.startswith("image/") and content_type != "application/octet-stream":
        logger.warning(f"[FULL VALIDATION] ❌ Invalid {label}: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be an image file. Got: {content_type}"
        )


def validate_image_integrity_full(image_bytes: bytes, label: str):
    """Ensure image bytes are valid and can be opened/verified by PIL."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verifies file integrity
    except Exception as e:
        logger.warning(f"[FULL VALIDATION] ❌ Invalid image bytes for {label}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} is not a valid image file or is corrupted."
        )



# ──────────────────────────────────────────────────────────────
# VISION ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────

async def analyze_full_outfit_with_vision(user_photo_bytes: bytes, outfit_photo_bytes: bytes) -> dict:
    """Analyze user's gender and full-body outfit photo details (dress, suit, jumpsuit, etc.)"""
    logger.info(f"[VISION] Analyzing user & full outfit photos with {VISION_MODEL}...")
    user_b64 = image_to_base64_full(user_photo_bytes)
    outfit_b64 = image_to_base64_full(outfit_photo_bytes)

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{user_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{outfit_b64}"}},
                    {
                        "type": "text",
                        "text": (
                            "Analyze these two photos for a virtual try-on:\n"
                            "The first image is the user's full-body photo.\n"
                            "The second image is the full outfit / dress / suit photo.\n"
                            "Please return a JSON object with the following fields:\n"
                            "1. 'gender': Determine the gender of the user ('man', 'woman', or 'person').\n"
                            "2. 'dress_type': e.g., 'dress', 'suit', 'jumpsuit', 'gown', 'outfit'.\n"
                            "3. 'description': Detailed description of the outfit (color, patterns, fabric style, neckline, sleeves, pockets, length, etc.) for try-on editing.\n\n"
                            "Provide the response in raw JSON format matching this schema:\n"
                            "{\"gender\": string, \"dress_type\": string, \"description\": string}"
                        )
                    }
                ]
            }],
            max_tokens=400
        )
        data = json.loads(response.choices[0].message.content)
        data.setdefault("gender", "person")
        data.setdefault("dress_type", "outfit")
        data.setdefault("description", "clothing outfit")
        return data
    except Exception as e:
        logger.error(f"[VISION ERROR] Full outfit analysis failed: {str(e)}")
        return {"gender": "person", "dress_type": "outfit", "description": "clothing outfit"}


async def analyze_shirt_with_vision(user_photo_bytes: bytes, shirt_photo_bytes: bytes) -> dict:
    """Analyze user's gender and shirt photo details."""
    logger.info(f"[VISION] Analyzing user & shirt photos with {VISION_MODEL}...")
    user_b64 = image_to_base64_full(user_photo_bytes)
    shirt_b64 = image_to_base64_full(shirt_photo_bytes)

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{user_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{shirt_b64}"}},
                    {
                        "type": "text",
                        "text": (
                            "Analyze these two photos for a virtual try-on:\n"
                            "The first image is the user's photo.\n"
                            "The second image is the shirt / top photo.\n"
                            "Please return a JSON object with the following fields:\n"
                            "1. 'gender': Determine the gender of the user ('man', 'woman', or 'person').\n"
                            "2. 'dress_type': e.g., 'shirt', 't-shirt', 'top', 'hoodie', 'sweater'.\n"
                            "3. 'description': Detailed description of the shirt (color, collar, sleeves, logo, pattern, buttons, pockets, fit) for try-on editing.\n\n"
                            "Provide the response in raw JSON format matching this schema:\n"
                            "{\"gender\": string, \"dress_type\": string, \"description\": string}"
                        )
                    }
                ]
            }],
            max_tokens=400
        )
        data = json.loads(response.choices[0].message.content)
        data.setdefault("gender", "person")
        data.setdefault("dress_type", "t-shirt")
        data.setdefault("description", "clothing")
        return data
    except Exception as e:
        logger.error(f"[VISION ERROR] Shirt analysis failed: {str(e)}")
        return {"gender": "person", "dress_type": "t-shirt", "description": "clothing"}


async def analyze_pant_with_vision(user_photo_bytes: bytes, pant_photo_bytes: bytes) -> dict:
    """Analyze user's gender and pant photo details."""
    logger.info(f"[VISION] Analyzing user & pant photos with {VISION_MODEL}...")
    user_b64 = image_to_base64_full(user_photo_bytes)
    pant_b64 = image_to_base64_full(pant_photo_bytes)

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{user_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pant_b64}"}},
                    {
                        "type": "text",
                        "text": (
                            "Analyze these two photos for a virtual try-on:\n"
                            "The first image is the user's photo.\n"
                            "The second image is the pant / shorts photo.\n"
                            "Please return a JSON object with the following fields:\n"
                            "1. 'gender': Determine the gender of the user ('man', 'woman', or 'person').\n"
                            "2. 'pant_type': e.g., 'pants', 'shorts', 'jeans', 'trousers', 'leggings'.\n"
                            "3. 'description': Detailed description of the pant (color, fabric denim/cotton, style, length, waistband, pockets) for try-on editing.\n\n"
                            "Provide the response in raw JSON format matching this schema:\n"
                            "{\"gender\": string, \"pant_type\": string, \"description\": string}"
                        )
                    }
                ]
            }],
            max_tokens=400
        )
        data = json.loads(response.choices[0].message.content)
        data.setdefault("gender", "person")
        data.setdefault("pant_type", "pants")
        data.setdefault("description", "pants")
        return data
    except Exception as e:
        logger.error(f"[VISION ERROR] Pant analysis failed: {str(e)}")
        return {"gender": "person", "pant_type": "pants", "description": "pants"}


# ──────────────────────────────────────────────────────────────
# CORE SWAP LOGIC
# ──────────────────────────────────────────────────────────────

async def execute_clothing_swap(
    user_photo_resized: bytes,
    prompt: str,
    output_prefix: str
) -> bytes:
    """Helper to perform OpenAI edit swap on image bytes and return result bytes."""
    user_img_io = io.BytesIO(user_photo_resized)
    user_img_io.name = "user_photo.png"

    response = await client.images.edit(
        model=OPENAI_MODEL,
        image=user_img_io,
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    generated_image_b64 = response.data[0].b64_json
    if not generated_image_b64:
        raise Exception("No image data returned from OpenAI")

    return base64.b64decode(generated_image_b64)


# ──────────────────────────────────────────────────────────────
# ROUTE
# ──────────────────────────────────────────────────────────────

@router.post("/api/full-try-on")
async def full_try_on(request: Request):
    """
    Virtual Full-Body Outfit / Double Try-On Endpoint.

    Supports two modes:
    1. Single Dress Try-On: Upload user_photo + dress_photo (full dress/suit/gown)
    2. Double Try-On: Upload user_photo + shirt_photo (upper body) + pant_photo (lower body)
    """
    form = await request.form()
    logger.info(f"[DEBUG FULL] Form keys received: {list(form.keys())}")

    # Strip spaces from keys
    form_data = {k.strip(): v for k, v in form.items()}

    user_photo = form_data.get("user_photo")
    dress_photo = form_data.get("fulll_drss_photo") or form_data.get("full_dress") or form_data.get("dress_photo")
    shirt_photo = form_data.get("shirt_drss_photo") or form_data.get("shirt_photo")
    pant_photo = form_data.get("pant _dress_photo") or form_data.get("pant_dress_photo") or form_data.get("pant_photo")
    
    shirt_size = form_data.get("shirt_dres_size") or form_data.get("dress_size") or form_data.get("size") or "M"
    pant_size = form_data.get("pant_dress_size") or form_data.get("dress_size") or form_data.get("size") or "M"

    if not user_photo or not getattr(user_photo, "filename", None):
        raise HTTPException(status_code=400, detail="user_photo is required.")

    # Validate that either fulll_drss_photo OR (shirt_drss_photo and/or pant_dress_photo) is provided
    has_dress = dress_photo and getattr(dress_photo, "filename", None)
    has_shirt = shirt_photo and getattr(shirt_photo, "filename", None)
    has_pant = pant_photo and getattr(pant_photo, "filename", None)

    if not has_dress and not has_shirt and not has_pant:
        raise HTTPException(
            status_code=400,
            detail="Either 'fulll_drss_photo' OR at least one of ('shirt_drss_photo', 'pant _dress_photo') is required."
        )

    # Validate sizes: allows standard letter sizes or any numeric size
    shirt_size_upper = str(shirt_size).strip().upper()
    if shirt_size_upper not in VALID_SIZES and not shirt_size_upper.isdigit():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{shirt_size}'. Valid: {', '.join(VALID_SIZES)} or any numeric size"
        )

    pant_size_upper = str(pant_size).strip().upper()
    if pant_size_upper not in VALID_SIZES and not pant_size_upper.isdigit():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{pant_size}'. Valid: {', '.join(VALID_SIZES)} or any numeric size"
        )

    # Validate content-types
    validate_image_full(user_photo, "User photo")
    if has_dress:
        validate_image_full(dress_photo, "Dress photo")
    if has_shirt:
        validate_image_full(shirt_photo, "Shirt photo")
    if has_pant:
        validate_image_full(pant_photo, "Pant photo")

    # Read bytes
    logger.info("[READ FULL] Reading image bytes...")
    user_bytes = await user_photo.read()
    if len(user_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="User photo exceeds 10MB limit")
    validate_image_integrity_full(user_bytes, "User photo")

    user_photo_resized = resize_image_full(user_bytes, max_size=1024)

    # Define variables to hold intermediate & final results
    result_bytes = None
    filename_out = ""

    try:
        # Mode 1: Single full dress/suit try-on
        if has_dress:
            logger.info("[MODE] Mode 1: Single Dress Outfit Try-On active")
            dress_bytes = await dress_photo.read()
            if len(dress_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="Dress photo exceeds 10MB limit")
            validate_image_integrity_full(dress_bytes, "Dress photo")
            
            dress_photo_resized = resize_image_full(dress_bytes, max_size=1024)

            # Analyze full outfit details
            analysis = await analyze_full_outfit_with_vision(user_photo_resized, dress_photo_resized)
            gender = analysis["gender"]
            dress_type = analysis["dress_type"]
            dress_desc = analysis["description"]

            logger.info(f"[ANALYSIS FULL] Gender: {gender} | Type: {dress_type} | Desc: {dress_desc[:100]}...")

            prompt = (
                f"Replace the entire upper and lower body clothing of the {gender} with ONLY this exact outfit: {dress_desc}. "
                f"Remove any existing shirts, jackets, pants, jeans, skirts, cardigans, sweaters, or other clothing layers the person is wearing. "
                f"The {gender} must wear ONLY this new outfit directly on their body, with no other layers visible. "
                f"STRICT RULES — DO NOT change anything else: "
                f"- Face, hair, skin tone, hands, and head: MUST remain pixel-perfect identical. "
                f"- Body position and pose: MUST remain exactly the same. "
                f"- Background and lighting: MUST remain exactly the same. "
                f"- The new outfit must fit naturally on the body for size {shirt_size_upper}."
            )
            logger.info("[STEP 3] Calling OpenAI edit for full outfit...")
            result_bytes = await execute_clothing_swap(user_photo_resized, prompt, "full")
            filename_out = f"full_tryon_{uuid.uuid4().hex[:12]}.png"

        # Mode 2: Double Try-on (Shirt + Pant swap)
        else:
            logger.info("[MODE] Mode 2: Multi-Garment Try-On active")
            current_user_image = user_photo_resized

            # Stage 1: Upper body shirt swap (if shirt_photo provided)
            if has_shirt:
                logger.info("[STAGE 1] Upper body shirt swap starting...")
                shirt_bytes = await shirt_photo.read()
                if len(shirt_bytes) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="Shirt photo exceeds 10MB limit")
                validate_image_integrity_full(shirt_bytes, "Shirt photo")
                
                shirt_photo_resized = resize_image_full(shirt_bytes, max_size=1024)

                shirt_analysis = await analyze_shirt_with_vision(current_user_image, shirt_photo_resized)
                gender = shirt_analysis["gender"]
                dress_type = shirt_analysis["dress_type"]
                shirt_desc = shirt_analysis["description"]

                logger.info(f"[ANALYSIS SHIRT] Gender: {gender} | Type: {dress_type} | Desc: {shirt_desc[:100]}...")

                shirt_prompt = (
                    f"Replace the entire upper body clothing of the {gender} with ONLY this exact top/shirt: {shirt_desc}. "
                    f"Remove any jackets, sweaters, coats, cardigans, or outer layers. "
                    f"The {gender} must wear ONLY this new top directly on their torso, with no other outer layers visible. "
                    f"STRICT RULES — DO NOT change anything else: "
                    f"- Face, hair, skin tone, head: MUST remain pixel-perfect identical. "
                    f"- Lower body clothing (pants/skirt): MUST remain exactly the same. "
                    f"- Body position and pose: MUST remain exactly the same. "
                    f"- Background and lighting: MUST remain exactly the same. "
                    f"- The new shirt must fit naturally on the body for size {shirt_size_upper}."
                )
                logger.info("[STAGE 1] Calling OpenAI edit for shirt...")
                current_user_image = await execute_clothing_swap(current_user_image, shirt_prompt, "shirt")

            # Stage 2: Lower body pant swap (if pant_photo provided)
            if has_pant:
                logger.info("[STAGE 2] Lower body pant swap starting...")
                pant_bytes = await pant_photo.read()
                if len(pant_bytes) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="Pant photo exceeds 10MB limit")
                validate_image_integrity_full(pant_bytes, "Pant photo")
                
                pant_photo_resized = resize_image_full(pant_bytes, max_size=1024)

                pant_analysis = await analyze_pant_with_vision(current_user_image, pant_photo_resized)
                gender = pant_analysis["gender"]
                pant_type = pant_analysis["pant_type"]
                pant_desc = pant_analysis["description"]

                logger.info(f"[ANALYSIS PANT] Gender: {gender} | Type: {pant_type} | Desc: {pant_desc[:100]}...")

                pant_prompt = (
                    f"Replace the entire lower body clothing of the {gender} with ONLY this exact bottom/pant: {pant_desc}. "
                    f"Remove any existing skirts, pants, shorts, or lower body layers. "
                    f"The {gender} must wear ONLY this new bottom. "
                    f"STRICT RULES — DO NOT change anything else: "
                    f"- Face, hair, skin tone, head: MUST remain pixel-perfect identical. "
                    f"- Upper body clothing (including the newly added shirt): MUST remain exactly the same. "
                    f"- Body position and pose: MUST remain exactly the same. "
                    f"- Background and lighting: MUST remain exactly the same. "
                    f"- The new pant must fit naturally on the body for size {pant_size_upper}."
                )
                logger.info("[STAGE 2] Calling OpenAI edit for pant...")
                current_user_image = await execute_clothing_swap(current_user_image, pant_prompt, "pant")

            result_bytes = current_user_image
            filename_out = f"multi_tryon_{uuid.uuid4().hex[:12]}.png"

        # Save result to outputs/
        file_path = OUTPUT_DIR / filename_out
        with open(file_path, "wb") as f:
            f.write(result_bytes)

        logger.info(f"[SAVED] Result saved to {file_path}")

        base_url = str(request.base_url)
        dynamic_url = f"{base_url}outputs/{filename_out}"

        return JSONResponse(
            status_code=200,
            content={
                "image_url": dynamic_url
            }
        )

    except openai.RateLimitError as e:
        logger.error(f"[RATE LIMIT] ❌ Rate limit exceeded: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail="AI service rate limit exceeded. Please wait a moment before trying again."
        )
    except openai.APIConnectionError as e:
        logger.error(f"[CONNECTION ERROR] ❌ Connection error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail="Could not reach the AI service provider. Please verify network connectivity."
        )
    except openai.APIStatusError as e:
        logger.error(f"[API ERROR] ❌ OpenAI status error {e.status_code}: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail=f"AI service error: {e.message}"
        )
    except Exception as e:
        logger.error(f"[FULL ERROR] Virtual try-on process failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Virtual try-on process failed: {str(e)}"
        )

