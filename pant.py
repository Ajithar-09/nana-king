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
from moderation import moderate_images

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
ALLOWED_TYPES  = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE  = 10 * 1024 * 1024  # 10 MB
VALID_SIZES    = ["XS", "S", "M", "L", "XL", "XXL", "XXXL",
                  "28", "30", "32", "34", "36", "38", "40", "42"]
VALID_TYPES    = ["pants", "shorts", "jeans", "trousers", "chinos",
                  "joggers", "track pants", "cargo pants", "leggings"]


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def resize_image_pant(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Resize image keeping aspect ratio, convert to PNG."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[PANT RESIZE] {w}x{h} → {new_w}x{new_h}")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def image_to_base64_pant(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_image_pant(file: UploadFile, label: str):
    """Accept any image format — Pillow handles conversion internally."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/") and content_type != "application/octet-stream":
        logger.warning(f"[PANT VALIDATION] ❌ Invalid {label}: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be an image file. Got: {content_type}"
        )
    logger.info(f"[PANT VALIDATION] ✅ {label} | Type: {content_type} | Name: {file.filename}")


def validate_image_integrity_pant(image_bytes: bytes, label: str):
    """Ensure image bytes are valid and can be opened/verified by PIL."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verifies file integrity
    except Exception as e:
        logger.warning(f"[PANT VALIDATION] ❌ Invalid image bytes for {label}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} is not a valid image file or is corrupted."
        )



async def analyze_photos_with_vision_pant(user_photo_bytes: bytes, pant_photo_bytes: bytes) -> dict:
    """
    Use Vision Model to analyze both the user photo and the pant/shorts photo.
    Returns a dictionary containing 'gender', 'pant_type', 'photo_type', and 'description'.
    """
    logger.info(f"[VISION] Analyzing user & pant photos with {VISION_MODEL}...")
    user_b64 = image_to_base64_pant(user_photo_bytes)
    pant_b64 = image_to_base64_pant(pant_photo_bytes)

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{user_b64}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{pant_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze these two photos for a virtual try-on and safety compliance:\n"
                            "The first image is the user's photo.\n"
                            "The second image is the pant / shorts / lower-body clothing photo.\n"
                            "First, perform a safety check on both photos. Check for:\n"
                            "- Nudity, semi-nudity (underwear, swimwear, or bare chests where inappropriate),\n"
                            "- Transparent or see-through clothing that reveals private body parts,\n"
                            "- Obscene/vulgar gestures (e.g., middle finger),\n"
                            "- Offensive/vulgar text, graphics, or symbols on clothing or background.\n\n"
                            "Please return a JSON object with the following fields:\n"
                            "1. 'is_safe': A boolean (true or false). Set to false if either image contains nudity, semi-nudity, underwear, transparent clothing, obscene gestures, or vulgar graphics/text. Otherwise, set to true.\n"
                            "2. 'safety_reason': If 'is_safe' is false, write a short descriptive reason in English explaining which image was unsafe and why. If safe, set to empty string.\n"
                            "3. 'gender': Determine the gender of the user from the first photo. It must be one of: 'man', 'woman', or 'person'.\n"
                            "4. 'pant_type': Determine the type of lower-body clothing in the second photo. E.g. 'pants', 'shorts', 'jeans', 'trousers', 'leggings', 'skirt'.\n"
                            "5. 'photo_type': Detect if the first photo is a full-body photo ('full') or a bottom-half/waist-down photo ('bottom').\n"
                            "6. 'description': A precise detailed description of the pant in the second photo (exact color, patterns, fabric like denim/cotton, waistband like elastic/drawstring/button, pockets, leg style like slim/straight/wide, and fit style for try-on editing).\n\n"
                            "Provide the response in raw JSON format matching this schema:\n"
                            "{\"is_safe\": boolean, \"safety_reason\": string, \"gender\": string, \"pant_type\": string, \"photo_type\": string, \"description\": string}"
                        )
                    }
                ]
            }],
            max_tokens=400
        )

        result_text = response.choices[0].message.content
        logger.info(f"[VISION] Analysis response: {result_text}")
        data = json.loads(result_text)
        
        # Ensure fallbacks
        data.setdefault("is_safe", True)
        data.setdefault("safety_reason", "")
        data.setdefault("gender", "person")
        data.setdefault("pant_type", "pants")
        data.setdefault("photo_type", "full")
        data.setdefault("description", "pants")
        
        if not data["is_safe"]:
            logger.warning(f"[VISION SAFETY] ❌ Safety check failed: {data['safety_reason']}")
            raise HTTPException(
                status_code=400,
                detail=f"Safety restriction: {data['safety_reason']}"
            )
            
        if data["gender"] not in ("man", "woman", "person"):
            data["gender"] = "person"
        if data["photo_type"] not in ("full", "bottom"):
            data["photo_type"] = "full"
            
        return data

    except HTTPException as e:
        # Propagate custom moderation/safety errors
        raise e
    except Exception as e:
        logger.error(f"[VISION ERROR] Vision analysis failed, using fallbacks: {str(e)}")
        return {
            "is_safe": True,
            "safety_reason": "",
            "gender": "person",
            "pant_type": "pants",
            "photo_type": "full",
            "description": "pants"
        }


async def generate_pant_tryon(
    user_photo_bytes: bytes,
    pant_photo_bytes: bytes,
    pant_size: str,
    base_url: str = "http://localhost:8000/"
) -> dict:
    """
    Generate virtual pant/shorts try-on by editing user's photo.

    Steps:
      1. Resize both images
      2. Analyze photos using Vision model → detailed description, pant type, photo type, gender
      3. Use images.edit() on user's photo targeting lower body only
      4. Save generated image to outputs/ folder
      5. Return image URL and base64
    """
    logger.info(f"[PANT TRY-ON START] Size: {pant_size}")

    # ── Step 1: Resize images ──────────────────────────────────
    logger.info("[STEP 1] Resizing images...")
    user_photo_resized = resize_image_pant(user_photo_bytes,  max_size=1024)
    pant_photo_resized = resize_image_pant(pant_photo_bytes, max_size=1024)

    # ── Step 2: Analyze photos with Vision model ───────────────
    logger.info("[STEP 2] Analyzing photos with GPT-4o Vision...")
    analysis = await analyze_photos_with_vision_pant(user_photo_resized, pant_photo_resized)
    gender = analysis["gender"]
    pant_type = analysis["pant_type"]
    photo_type = analysis["photo_type"]
    pant_description = analysis["description"]

    logger.info(f"[ANALYSIS DETECTED] Gender: {gender} | Type: {pant_type} | Photo Type: {photo_type} | Desc: {pant_description[:100]}...")

    # ── Step 3: Build edit prompt ──────────────────────────────
    # Adjust prompt based on whether user sent full body or bottom-half photo
    if photo_type == "full":
        body_area = "lower body (waist down) — the legs and hip area only"
        keep_same = "Face, hair, upper body clothing, torso, background, and lighting"
    else:
        body_area = "entire lower body area visible in this photo"
        keep_same = "Waistline, body shape, background, and lighting"

    prompt = (
        f"Minimal bottom clothing swap only. "
        f"In this photo, change ONLY the {pant_type} on the {gender}'s {body_area}. "
        f"Replace it with this exact {pant_type}: {pant_description}. "
        f"STRICT RULES — DO NOT change anything else: "
        f"- {keep_same}: MUST remain pixel-perfect identical. "
        f"- Body position and pose: MUST remain exactly the same. "
        f"- Only the {pant_type} fabric/color/pattern in the lower body changes. "
        f"- The new {pant_type} must fit naturally on the body for size {pant_size}. "
        f"- Preserve natural wrinkles, shadows, and folds consistent with original photo. "
        f"- Result must look like the original photo with only the {pant_type} swapped."
    )
    logger.info(f"[STEP 3] Edit prompt built | Length: {len(prompt)} chars")

    # ── Step 4: Call images.edit() on user's real photo ───────
    logger.info(f"[STEP 4] Calling images.edit() with model: {OPENAI_MODEL}...")

    user_img_io = io.BytesIO(user_photo_resized)
    user_img_io.name = "user_photo.png"

    response = await client.images.edit(
        model=OPENAI_MODEL,
        image=user_img_io,
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    logger.info("[STEP 4] OpenAI edit response received successfully")

    # ── Step 5: Extract generated image ───────────────────────
    generated_image_b64 = response.data[0].b64_json

    if not generated_image_b64:
        logger.warning("[STEP 5] No b64_json in response")
        raise Exception("No image data returned from OpenAI")

    logger.info(f"[STEP 5] Image extracted | Base64 length: {len(generated_image_b64)}")

    # ── Step 6: Save image to outputs/ folder ─────────────────
    filename        = f"pant_{uuid.uuid4().hex[:12]}.png"
    file_path       = OUTPUT_DIR / filename
    image_bytes_out = base64.b64decode(generated_image_b64)
    with open(file_path, "wb") as f:
        f.write(image_bytes_out)

    image_url = f"{base_url}outputs/{filename}"

    logger.info(f"[STEP 6] Image saved → {file_path}")
    logger.info(f"[STEP 6] Access URL  → {image_url}")
    logger.info("[PANT TRY-ON COMPLETE] ✅ Pant/shorts try-on complete!")

    return {
        "success":      True,
        "image_base64": generated_image_b64,
        "image_url":    image_url,
        "filename":     filename,
        "message":      f"Pant try-on generated! Size: {pant_size}, Type: {pant_type}"
    }


# ──────────────────────────────────────────────────────────────
# ROUTE
# ──────────────────────────────────────────────────────────────

@router.post("/api/pant-try-on")
async def pant_try_on(request: Request):
    """
    Virtual Pant/Shorts Try-On Endpoint.

    - Upload user's photo (full body OR bottom half)
    - Upload pants/shorts photo
    - AI analyzes the pants automatically → edits only the lower body in user's real photo
    - Returns user's real photo with the pants/shorts applied
    """
    form = await request.form()
    logger.info(f"[DEBUG] Raw Form Keys: {list(form.keys())}")

    # Strip spaces from keys
    form_data = {k.strip(): v for k, v in form.items()}

    user_photo = form_data.get("user_photo")
    pant_photo = form_data.get("dress_photo") or form_data.get("pant_photo")
    pant_size = form_data.get("dress_size") or form_data.get("pant_size")

    if not user_photo or not getattr(user_photo, "filename", None):
        logger.warning(f"[VALIDATION] ❌ Missing user photo. Available keys: {list(form_data.keys())}")
        raise HTTPException(
            status_code=400,
            detail="user_photo is required."
        )

    if not pant_photo or not getattr(pant_photo, "filename", None):
        logger.warning(f"[VALIDATION] ❌ Missing dress photo. Available keys: {list(form_data.keys())}")
        raise HTTPException(
            status_code=400,
            detail="dress_photo is required."
        )

    if not pant_size:
        logger.warning(f"[VALIDATION] ❌ Missing size. Available keys: {list(form_data.keys())}")
        raise HTTPException(
            status_code=400,
            detail="dress_size is required."
        )

    pant_size_str = str(pant_size).strip()

    logger.info("=" * 60)
    logger.info("[PANT REQUEST] 📥 New pant try-on request")
    logger.info(f"  User Photo : {user_photo.filename}")
    logger.info(f"  Dress Photo: {pant_photo.filename}")
    logger.info(f"  Dress Size : {pant_size_str}")
    logger.info("=" * 60)

    # Validate size: allows standard sizes (XS, S, M, L, XL, XXL, XXXL) or any numeric size
    size_upper = pant_size_str.upper()
    if size_upper not in VALID_SIZES and not size_upper.isdigit():
        logger.warning(f"[VALIDATION] ❌ Invalid size: {pant_size_str}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{pant_size_str}'. Valid: {', '.join(VALID_SIZES)} or any numeric size (e.g. 28, 30, 32, 34, 36, 38, 40, 42)"
        )

    # Validate content-type headers
    validate_image_pant(user_photo, "User photo")
    validate_image_pant(pant_photo, "Pant photo")

    # Read bytes
    logger.info("[READ] Reading uploaded image bytes...")
    user_photo_bytes = await user_photo.read()
    pant_photo_bytes = await pant_photo.read()

    # Check file sizes
    if len(user_photo_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="User photo exceeds 10MB limit")
    if len(pant_photo_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Pant photo exceeds 10MB limit")

    # Validate image bytes integrity
    validate_image_integrity_pant(user_photo_bytes, "User photo")
    validate_image_integrity_pant(pant_photo_bytes, "Pant photo")

    # Content moderation (pre-validation)
    await moderate_images(
        client=client,
        images_bytes=[user_photo_bytes, pant_photo_bytes],
        labels=["User photo", "Pant photo"]
    )

    logger.info(f"[READ] User photo : {len(user_photo_bytes) / 1024:.1f} KB")
    logger.info(f"[READ] Pant photo : {len(pant_photo_bytes) / 1024:.1f} KB")

    base_url = str(request.base_url)

    # Generate try-on
    try:
        result = await generate_pant_tryon(
            user_photo_bytes=user_photo_bytes,
            pant_photo_bytes=pant_photo_bytes,
            pant_size=size_upper,
            base_url=base_url
        )

        logger.info("[RESPONSE] ✅ Sending pant try-on success response")
        return JSONResponse(
            status_code=200,
            content={
                "image_url": result["image_url"]
            }
        )

    except HTTPException as e:
        # Propagate custom moderation / validation errors
        raise e
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
        logger.error(f"[PANT ERROR] ❌ Pant try-on failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pant try-on failed: {str(e)}")

