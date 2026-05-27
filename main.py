import os
import logging
import uvicorn
from dotenv import load_dotenv
from shirt import app
load_dotenv()

logger = logging.getLogger(__name__)


# ─── Health Check ──────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    logger.info("[ROUTE] GET /api/health")
    api_key = os.getenv("OPENAI_API_KEY", "")
    key_ok  = bool(api_key and api_key != "your_openai_api_key_here")
    return {
        "status": "healthy",
        "openai_api_key": "✅ configured" if key_ok else "❌ not configured",
        "model": os.getenv("OPENAI_MODEL"),
        "port":  os.getenv("PORT")
    }


# ─── Run Server ────────────────────────────────────────────────
if __name__ == "__main__":
    port_env = os.getenv("PORT")
    port     = int(port_env) if port_env else 8000
    model    = os.getenv("OPENAI_MODEL")
    logger.info("🚀 Starting Virtual Try-On API Server...")
    logger.info(f"📍 Port  : {port}")
    logger.info(f"🤖 Model : {model}")
    logger.info(f"📖 Docs  : http://localhost:{port}/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
