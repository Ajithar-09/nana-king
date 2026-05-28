import base64
import logging
from fastapi import HTTPException
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

def image_to_base64_mod(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

async def moderate_images(client: AsyncOpenAI, images_bytes: list[bytes], labels: list[str]) -> None:
    """
    Moderates a list of images in a single API call to OpenAI's omni-moderation-latest.
    Raises HTTPException (400) if any image is flagged.
    """
    if not images_bytes:
        return

    logger.info(f"[MODERATION] Moderating {len(images_bytes)} images against omni-moderation-latest...")
    
    # Prepare the multimodal inputs
    inputs = []
    for img_bytes in images_bytes:
        b64_str = image_to_base64_mod(img_bytes)
        inputs.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_str}"
            }
        })

    try:
        response = await client.moderations.create(
            model="omni-moderation-latest",
            input=inputs
        )
        
        for i, result in enumerate(response.results):
            label = labels[i] if i < len(labels) else f"Image {i+1}"
            if result.flagged:
                # Retrieve flagged categories
                flagged_categories = [cat for cat, val in result.categories.model_dump().items() if val]
                logger.warning(f"[MODERATION] ❌ {label} flagged for: {flagged_categories}")
                
                # Format categories for clean reporting
                categories_str = ", ".join(flagged_categories)
                raise HTTPException(
                    status_code=400,
                    detail=f"{label} violates safety policies (flagged: {categories_str}). Please upload appropriate images."
                )
        
        logger.info("[MODERATION] ✅ All images passed content moderation.")

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"[MODERATION ERROR] OpenAI Moderation API failed: {str(e)}")
        # Fallback: Let it pass through to the secondary GPT-4o Vision safety check.
        # This prevents absolute downtime if the free moderation endpoint is degraded.
        pass
