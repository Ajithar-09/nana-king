import os
import base64
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# ─── Logger ───────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Config from .env ─────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")
VISION_MODEL   = os.getenv("VISION_MODEL")

# ─── OpenAI Client ────────────────────────────────────────────
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ─── Output Folder ────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
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
    if not content_type.startswith("image/"):
        logger.warning(f"[PANT VALIDATION] ❌ Invalid {label}: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be an image file. Got: {content_type}"
        )
    logger.info(f"[PANT VALIDATION] ✅ {label} | Type: {content_type} | Name: {file.filename}")


async def analyze_pant_with_vision(pant_photo_bytes: bytes, pant_type: str) -> str:
    """
    Use Vision model to analyze pant/shorts photo and return precise description.
    Used to accurately recreate the pant on user's photo.
    """
    logger.info(f"[VISION] Analyzing {pant_type} photo with {VISION_MODEL}...")
    pant_b64 = image_to_base64_pant(pant_photo_bytes)

    response = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pant_b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        f"Analyze this {pant_type} for a virtual try-on. "
                        f"Describe only the clothing details precisely: "
                        f"1) Exact color(s) "
                        f"2) Pattern (solid, striped, camouflage, graphic, etc.) "
                        f"3) Fabric appearance (denim, cotton, linen, polyester, etc.) "
                        f"4) Waistband style (elastic, drawstring, button, belt loops) "
                        f"5) Leg style (straight, slim, wide-leg, tapered, bootcut) "
                        f"6) Length (full length, knee-length, mid-thigh, ankle) "
                        f"7) Any pockets, zippers, logos, or special details "
                        f"8) Fit style (slim, regular, relaxed, oversized) "
                        f"Keep description short and precise."
                    )
                }
            ]
        }],
        max_tokens=300
    )

    description = response.choices[0].message.content
    logger.info(f"[VISION] {pant_type} analyzed | Description: {description[:100]}...")
    return description


async def generate_pant_tryon(
    user_photo_bytes: bytes,
    pant_photo_bytes: bytes,
    pant_size: str,
    pant_type: str,
    photo_type: str = "full",
    gender: str = "person"
) -> dict:
    """
    Generate virtual pant/shorts try-on by editing user's photo.

    Steps:
      1. Resize both images
      2. Analyze pant photo using Vision model → detailed description
      3. Use images.edit() on user's photo targeting lower body only
      4. Save generated image to outputs/ folder
      5. Return image URL and base64
    """
    logger.info(f"[PANT TRY-ON START] Type: {pant_type} | Size: {pant_size} | Photo: {photo_type} | Gender: {gender}")

    # ── Step 1: Resize images ──────────────────────────────────
    logger.info("[STEP 1] Resizing images...")
    user_photo_resized = resize_image_pant(user_photo_bytes,  max_size=1024)
    pant_photo_resized = resize_image_pant(pant_photo_bytes, max_size=1024)

    # ── Step 2: Analyze pant with Vision model ─────────────────
    logger.info("[STEP 2] Analyzing pant/shorts with Vision model...")
    pant_description = await analyze_pant_with_vision(pant_photo_resized, pant_type)

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

    port      = int(os.getenv("PORT", 8000))
    image_url = f"http://localhost:{port}/outputs/{filename}"

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
async def pant_try_on(
    user_photo:  UploadFile = File(..., description="User's photo — full body OR bottom half (waist down)"),
    pant_photo:  UploadFile = File(..., description="Pants / shorts / jeans photo"),
    pant_size:   str        = Form(..., description="Size: S, M, L, XL or waist 28, 30, 32..."),
    pant_type:   str        = Form(default="pants",  description="pants / shorts / jeans / trousers / chinos"),
    photo_type:  str        = Form(default="full",   description="full = full body photo | bottom = waist-down photo"),
    gender:      str        = Form(default="person", description="person / man / woman")
):
    """
    Virtual Pant/Shorts Try-On Endpoint.

    - Upload user's photo (full body OR bottom half)
    - Upload pants/shorts photo
    - AI analyzes the pants → edits only the lower body in user's real photo
    - Returns user's real photo with the pants/shorts applied
    """
    logger.info("=" * 60)
    logger.info("[PANT REQUEST] 📥 New pant try-on request")
    logger.info(f"  User Photo : {user_photo.filename}")
    logger.info(f"  Pant Photo : {pant_photo.filename}")
    logger.info(f"  Pant Type  : {pant_type}")
    logger.info(f"  Pant Size  : {pant_size}")
    logger.info(f"  Photo Type : {photo_type} ({'full body' if photo_type == 'full' else 'bottom half'})")
    logger.info(f"  Gender     : {gender}")
    logger.info("=" * 60)

    # Validate size
    if pant_size.upper() not in VALID_SIZES and pant_size not in VALID_SIZES:
        logger.warning(f"[VALIDATION] ❌ Invalid pant size: {pant_size}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{pant_size}'. Valid: {', '.join(VALID_SIZES)}"
        )

    # Validate photo_type
    if photo_type not in ("full", "bottom"):
        raise HTTPException(
            status_code=400,
            detail="photo_type must be 'full' (full body) or 'bottom' (waist-down photo)"
        )

    # Validate images
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

    logger.info(f"[READ] User photo : {len(user_photo_bytes) / 1024:.1f} KB")
    logger.info(f"[READ] Pant photo : {len(pant_photo_bytes) / 1024:.1f} KB")

    # Generate try-on
    try:
        result = await generate_pant_tryon(
            user_photo_bytes=user_photo_bytes,
            pant_photo_bytes=pant_photo_bytes,
            pant_size=pant_size.upper(),
            pant_type=pant_type,
            photo_type=photo_type,
            gender=gender
        )

        logger.info("[RESPONSE] ✅ Sending pant try-on success response")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "image_url":    result["image_url"],
                    "filename":     result["filename"],
                    "image_base64": result["image_base64"],
                    "pant_size":    pant_size.upper(),
                    "pant_type":    pant_type,
                    "photo_type":   photo_type,
                    "gender":       gender,
                    "message":      result["message"]
                }
            }
        )

    except Exception as e:
        logger.error(f"[PANT ERROR] ❌ Pant try-on failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pant try-on failed: {str(e)}")
