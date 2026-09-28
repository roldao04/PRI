import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from indexa.entrypoints.api.routes.healthcheck import router as healthcheck_router
from indexa.entrypoints.api.routes.search import get_reranker
from indexa.entrypoints.api.routes.search import router as search_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="My Search Engine", swagger_ui_parameters={"operationsSorter": "alpha"}, prefix="/api/v1"
)

# Get the static files directory path
STATIC_DIR = Path(__file__).parent.parent.parent.parent.parent / "static_pages"

# main API router
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(healthcheck_router)
api_router.include_router(search_router)

# app
app.include_router(api_router)


# Add route to serve the main static page
@app.get("/")
async def serve_index():
    """Serve the main static search page."""
    static_file = STATIC_DIR / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {"message": "Static page not found. Please ensure static_pages/index.html exists."}


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def preload_models():
    """Preload neural reranking models and embeddings at startup.

    This warms up the cache by loading:
    1. Neural reranking models:
       - unicamp-dl: Lightweight Portuguese model (default)
       - mmarco: Multilingual model (heavier but better quality)
    2. Document embeddings for semantic similarity search

    All resources are loaded asynchronously in the background to not block server startup.
    """
    import asyncio

    logger.info("🚀 Starting application resource preload...")

    async def load_model(model_name: str):
        """Load a single model asynchronously."""
        try:
            logger.info(f"Preloading reranker model: {model_name}")
            # Run blocking model loading in thread pool to avoid blocking startup
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, get_reranker, model_name)
            logger.info(f"Reranker model preloaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to preload reranker model {model_name}: {e}")

    async def load_embeddings():
        """Load embeddings asynchronously."""
        try:
            from indexa.entrypoints.api.routes.search import get_search_engine

            logger.info("Preloading document embeddings...")
            # Run blocking embedding loading in thread pool
            loop = asyncio.get_event_loop()
            engine = await loop.run_in_executor(None, get_search_engine)

            # Check if embeddings loaded successfully
            if engine._embeddings_available and engine.embedding_similarity is not None:
                info = engine.embedding_similarity.get_info()
                logger.info(
                    f"✅ Document embeddings preloaded successfully: "
                    f"{info['num_documents']:,} docs, "
                    f"{info['memory_usage_mb']:.2f} MB, "
                    f"mapping={info['has_mapping']}"
                )
            else:
                logger.warning(
                    "⚠️  Document embeddings not available. "
                    "Similarity search will not be available. "
                    "Generate embeddings using: python scripts/generate_embeddings.py"
                )
        except Exception as e:
            logger.error(f"Failed to preload embeddings: {e}")
            logger.warning("Similarity search features will not be available")

    # Preload all resources in parallel
    await asyncio.gather(
        load_model("unicamp-dl"),
        load_model("mmarco"),
        load_embeddings(),
        return_exceptions=True,  # Don't fail startup if loading fails
    )

    logger.info("✅ Resource preload complete - server ready")
