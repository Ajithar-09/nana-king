import os
import base64
import logging
import uuid
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image
import io
import json
from moderation import moderate_images

load_dotenv()

# ─── Logger Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Config from .env ─────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")
VISION_MODEL   = os.getenv("VISION_MODEL")

logger.info(f"[CONFIG] Image Model  : {OPENAI_MODEL}")
logger.info(f"[CONFIG] Vision Model : {VISION_MODEL}")

# ─── OpenAI Client ────────────────────────────────────────────
client = AsyncOpenAI(api_key=OPENAI_API_KEY or "placeholder_key_not_set")

# ─── Output Folder Setup ──────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
logger.info(f"[CONFIG] Output folder: {OUTPUT_DIR.resolve()}")

# ─── Background File Cleanup Task ─────────────────────────────
async def cleanup_old_files_task():
    """Background task running hourly to delete outputs older than 24 hours."""
    while True:
        try:
            logger.info("[CLEANUP] Scanning for old files in outputs/...")
            now = datetime.now()
            cutoff = now - timedelta(hours=24)
            count = 0
            if OUTPUT_DIR.exists():
                for file_path in OUTPUT_DIR.glob("*.png"):
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff:
                            file_path.unlink()
                            count += 1
            if count > 0:
                logger.info(f"[CLEANUP] Deleted {count} files older than 24 hours.")
            else:
                logger.info("[CLEANUP] No old files found.")
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] Failed to clean files: {e}")
        
        # Sleep for 1 hour
        await asyncio.sleep(3600)

# ─── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Virtual Try-On API",
    description="AI-powered virtual try-on — shirt & pant try-on on real user photo",
    version="2.0.0"
)

# ─── Startup Event ────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    # 1. Verify critical environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or "sk-" not in openai_key or openai_key == "placeholder_key_not_set":
        logger.error("❌ CRITICAL: OPENAI_API_KEY is not configured or is invalid! AI features will fail.")
    else:
        logger.info("✅ OPENAI_API_KEY is configured.")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        logger.error("❌ CRITICAL: OPENAI_MODEL is not set in environment.")
    else:
        logger.info(f"✅ Image Model: {model}")

    vision_model = os.getenv("VISION_MODEL")
    if not vision_model:
        logger.error("❌ CRITICAL: VISION_MODEL is not set in environment.")
    else:
        logger.info(f"✅ Vision Model: {vision_model}")

    # 2. Start background cleanup task
    asyncio.create_task(cleanup_old_files_task())
    logger.info("🧹 Background task cleanup_old_files_task initialized (hourly run).")

# ─── Static Files (serve saved images) ───────────────────────
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ─── CORS ─────────────────────────────────────────────────────
origins_env = os.getenv("CORS_ORIGINS")
origins = [origin.strip() for origin in origins_env.split(",")] if origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Include Pant Router ──────────────────────────────────────
from pant import router as pant_router
app.include_router(pant_router)

# ─── Include Full Router ──────────────────────────────────────
from full import router as full_router
app.include_router(full_router)

# ─── Constants ────────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
VALID_SIZES   = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def resize_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Resize image keeping aspect ratio, convert to PNG."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[IMAGE RESIZE] {w}x{h} → {new_w}x{new_h}")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_image(file: UploadFile, label: str):
    """Accept any image format — Pillow handles conversion internally."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/") and content_type != "application/octet-stream":
        logger.warning(f"[VALIDATION] ❌ Invalid {label} type: {content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be an image file. Got: {content_type}"
        )
    logger.info(f"[VALIDATION] ✅ {label} | Type: {content_type} | Name: {file.filename}")


def validate_image_integrity(image_bytes: bytes, label: str):
    """Ensure image bytes are valid and can be opened/verified by PIL."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verifies file integrity
    except Exception as e:
        logger.warning(f"[VALIDATION] ❌ Invalid image bytes for {label}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"{label} is not a valid image file or is corrupted."
        )



async def analyze_photos_with_vision(user_photo_bytes: bytes, dress_photo_bytes: bytes) -> dict:
    """
    Use Vision Model to analyze both the user photo and the dress photo.
    Returns a dictionary containing 'gender', 'dress_type', and 'description'.
    """
    logger.info(f"[VISION] Analyzing user & dress photos with {VISION_MODEL}...")
    user_b64 = image_to_base64(user_photo_bytes)
    dress_b64 = image_to_base64(dress_photo_bytes)

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
                        "image_url": {"url": f"data:image/png;base64,{dress_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze these two photos for a virtual try-on and safety compliance:\n"
                            "The first image is the user's photo.\n"
                            "The second image is the dress photo.\n"
                            "First, perform a safety check on both photos. Check for:\n"
                            "- Nudity, semi-nudity (underwear, swimwear, or bare chests where inappropriate),\n"
                            "- Transparent or see-through clothing that reveals private body parts,\n"
                            "- Obscene/vulgar gestures (e.g., middle finger),\n"
                            "- Offensive/vulgar text, graphics, or symbols on clothing or background.\n\n"
                            "Please return a JSON object with the following fields:\n"
                            "1. 'is_safe': A boolean (true or false). Set to false if either image contains nudity, semi-nudity, underwear, transparent clothing, obscene gestures, or vulgar graphics/text. Otherwise, set to true.\n"
                            "2. 'safety_reason': If 'is_safe' is false, write a short descriptive reason in English explaining which image was unsafe and why. If safe, set to empty string.\n"
                            "3. 'gender': Determine the gender of the user from the first photo. It must be one of: 'man', 'woman', or 'person'.\n"
                            "4. 'dress_type': Determine the type of clothing in the second photo. It must be a short name like 'shirt', 't-shirt', 'hoodie', 'sweater', 'top', etc.\n"
                            "5. 'description': A detailed description of the dress in the second photo (color, pattern, neckline, sleeve length, logos/prints, buttons/zippers, etc.) for try-on editing.\n\n"
                            "Provide the response in raw JSON format matching this schema:\n"
                            "{\"is_safe\": boolean, \"safety_reason\": string, \"gender\": string, \"dress_type\": string, \"description\": string}"
                        )
                    }
                ]
            }],
            max_tokens=400
        )

        result_text = response.choices[0].message.content
        logger.info(f"[VISION] Analysis response: {result_text}")
        data = json.loads(result_text)
        
        # Ensure fallback keys exist and are valid
        data.setdefault("is_safe", True)
        data.setdefault("safety_reason", "")
        data.setdefault("gender", "person")
        data.setdefault("dress_type", "t-shirt")
        data.setdefault("description", "clothing")
        
        if not data["is_safe"]:
            logger.warning(f"[VISION SAFETY] ❌ Safety check failed: {data['safety_reason']}")
            raise HTTPException(
                status_code=400,
                detail=f"Safety restriction: {data['safety_reason']}"
            )
            
        if data["gender"] not in ("man", "woman", "person"):
            data["gender"] = "person"
            
        return data

    except HTTPException as e:
        # Propagate safety HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"[VISION ERROR] Vision analysis failed, using fallbacks: {str(e)}")
        return {
            "is_safe": True,
            "safety_reason": "",
            "gender": "person",
            "dress_type": "t-shirt",
            "description": "clothing swap"
        }


async def generate_virtual_tryon(
    user_photo_bytes: bytes,
    dress_photo_bytes: bytes,
    dress_size: str,
    base_url: str = "http://localhost:8000/"
) -> dict:
    """
    Generate virtual try-on by editing the actual user photo to wear the given dress.

    Steps:
      1. Resize both images
      2. Analyze photos using GPT-4o Vision → gender, dress_type, dress_description
      3. Use images.edit() on user's real photo with dress description
      4. Save generated image to outputs/ folder
      5. Return image URL
    """
    logger.info(f"[TRY-ON START] Size: {dress_size}")

    # ── Step 1: Resize images ──────────────────────────────────
    logger.info("[STEP 1] Resizing images...")
    user_photo_resized  = resize_image(user_photo_bytes,  max_size=1024)
    dress_photo_resized = resize_image(dress_photo_bytes, max_size=1024)

    # ── Step 2: Analyze photos with GPT-4o Vision ──────────────
    logger.info("[STEP 2] Analyzing photos with GPT-4o Vision...")
    analysis = await analyze_photos_with_vision(user_photo_resized, dress_photo_resized)
    gender = analysis["gender"]
    dress_type = analysis["dress_type"]
    dress_description = analysis["description"]

    logger.info(f"[ANALYSIS DETECTED] Gender: {gender} | Type: {dress_type} | Desc: {dress_description[:100]}...")

    # ── Step 3: Build edit prompt ──────────────────────────────
    prompt = (
        f"Replace the entire upper body clothing of the {gender} with ONLY this exact {dress_type}: {dress_description}. "
        f"Remove any jackets, cardigans, sweaters, overcoats, coats, hoodies, or other outer layers the person is wearing. "
        f"The {gender} must wear ONLY the new {dress_type} directly on their torso, with no other outer layers or undergarments visible. "
        f"STRICT RULES — DO NOT change anything else: "
        f"- Face, hair, skin tone, hands: MUST remain pixel-perfect identical. "
        f"- Body position and pose: MUST remain exactly the same. "
        f"- Background and lighting: MUST remain exactly the same. "
        f"- The new {dress_type} must fit naturally on the body for size {dress_size}. "
        f"- Result must look like the original person wearing ONLY the new {dress_type}."
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
    filename  = f"tryon_{uuid.uuid4().hex[:12]}.png"
    file_path = OUTPUT_DIR / filename
    image_bytes_out = base64.b64decode(generated_image_b64)
    with open(file_path, "wb") as f:
        f.write(image_bytes_out)

    image_url = f"{base_url}outputs/{filename}"

    logger.info(f"[STEP 6] Image saved → {file_path}")
    logger.info(f"[STEP 6] Access URL  → {image_url}")
    logger.info("[TRY-ON COMPLETE] ✅ Virtual try-on complete — user's real photo edited with dress!")

    return {
        "success":      True,
        "image_base64": generated_image_b64,
        "image_url":    image_url,
        "filename":     filename,
        "message":      f"Try-on generated! Size: {dress_size}, Type: {dress_type}"
    }



# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint - API info."""
    logger.info("[ROUTE] GET / - API root accessed")
    return {
        "service": "Virtual Try-On API 👕",
        "version": "2.0.0",
        "model":   OPENAI_MODEL,
        "status":  "running",
        "endpoints": {
            "try_on": "POST /api/try-on",
            "health": "GET /api/health",
            "docs":   "GET /docs"
        }
    }


@app.post("/api/try-on")
async def virtual_try_on(
    request: Request,
    user_photo:  UploadFile = File(...,  description="User's half-body photo (upper body)"),
    dress_photo: UploadFile = File(...,  description="Dress / shirt / t-shirt photo"),
    dress_size:  str        = Form(...,  description="Dress size: S, M, L, XL, XXL")
):
    """
    Virtual Try-On Endpoint.

    - Upload user's real photo (half body)
    - Upload dress/shirt photo
    - AI analyzes the dress → edits user's actual photo to wear it
    - Returns user's real photo with the dress applied
    """
    logger.info("=" * 60)
    logger.info("[TRY-ON REQUEST] 📥 New request received")
    logger.info(f"  User Photo : {user_photo.filename}")
    logger.info(f"  Dress Photo: {dress_photo.filename}")
    logger.info(f"  Dress Size : {dress_size}")
    logger.info("=" * 60)

    # Validate size: allows standard sizes (XS, S, M, L, XL, XXL, XXXL) or any numeric size
    size_upper = dress_size.strip().upper()
    if size_upper not in VALID_SIZES and not size_upper.isdigit():
        logger.warning(f"[VALIDATION] ❌ Invalid dress size: {dress_size}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{dress_size}'. Valid: {', '.join(VALID_SIZES)} or any numeric size (e.g. 28, 30, 32, 34, 36, 38, 40, 42)"
        )

    # Validate content-type headers
    validate_image(user_photo,  "User photo")
    validate_image(dress_photo, "Dress photo")

    # Read bytes
    logger.info("[READ] Reading uploaded image bytes...")
    user_photo_bytes  = await user_photo.read()
    dress_photo_bytes = await dress_photo.read()

    # Check file sizes
    if len(user_photo_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="User photo exceeds 10MB limit")
    if len(dress_photo_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Dress photo exceeds 10MB limit")

    # Validate image bytes integrity
    validate_image_integrity(user_photo_bytes, "User photo")
    validate_image_integrity(dress_photo_bytes, "Dress photo")

    # Content moderation (pre-validation)
    await moderate_images(
        client=client,
        images_bytes=[user_photo_bytes, dress_photo_bytes],
        labels=["User photo", "Dress photo"]
    )

    logger.info(f"[READ] User photo  : {len(user_photo_bytes)  / 1024:.1f} KB")
    logger.info(f"[READ] Dress photo : {len(dress_photo_bytes) / 1024:.1f} KB")

    base_url = str(request.base_url)

    # Generate try-on
    try:
        result = await generate_virtual_tryon(
            user_photo_bytes=user_photo_bytes,
            dress_photo_bytes=dress_photo_bytes,
            dress_size=size_upper,
            base_url=base_url
        )

        logger.info("[RESPONSE] ✅ Sending success response to client")
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
        logger.error(f"[ERROR] ❌ Try-on generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Try-on generation failed: {str(e)}")

