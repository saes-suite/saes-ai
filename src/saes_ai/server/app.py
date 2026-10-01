"""FastAPI standalone server for SAES-AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saes_ai.server.routes import router as chat_router


def create_app() -> FastAPI:
    """Create and configure standalone SAES-AI FastAPI application."""
    app = FastAPI(
        title="SAES-AI REST API",
        description="Stochastic Algorithm Evaluation Suite - AI Intelligence & Conversational Assistant Server",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "saes-ai-api", "version": "0.1.0"}

    app.include_router(chat_router)
    return app


app = create_app()


def main() -> None:
    """CLI entry point for running the standalone SAES-AI REST API server."""
    import uvicorn

    uvicorn.run("saes_ai.server.app:app", host="0.0.0.0", port=8001, reload=True)


if __name__ == "__main__":
    main()
