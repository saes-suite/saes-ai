"""FastAPI standalone server for SAES-AI."""

import logging
import os
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from saes_ai.server.routes import router as chat_router

logger = logging.getLogger("saes_ai.server")


def configure_logging(level: str | None = None) -> None:
    """Configure structured console logging for the ``saes_ai`` package."""
    resolved = (level or os.getenv("SAES_AI_LOG_LEVEL") or "INFO").upper().strip()
    numeric_level = getattr(logging, resolved, logging.INFO)

    pkg_logger = logging.getLogger("saes_ai")
    pkg_logger.setLevel(numeric_level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        pkg_logger.addHandler(handler)
    for h in pkg_logger.handlers:
        h.setLevel(numeric_level)
    pkg_logger.propagate = False


def create_app() -> FastAPI:
    """Create and configure standalone SAES-AI FastAPI application."""
    from saes_ai._version import __version__

    configure_logging()

    app = FastAPI(
        title="SAES-AI REST API",
        description="Stochastic Algorithm Evaluation Suite - AI Intelligence & Conversational Assistant Server",
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not logger.isEnabledFor(logging.DEBUG):
            return await call_next(request)
        t0 = time.perf_counter()
        logger.debug("HTTP --> %s %s", request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "HTTP <-- %s %s status=%d elapsed=%.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "saes-ai-api", "version": __version__}

    app.include_router(chat_router)
    return app


app = create_app()


def main() -> None:
    """CLI entry point for running the standalone SAES-AI REST API server."""
    import uvicorn

    log_level = (os.getenv("SAES_AI_LOG_LEVEL") or "info").lower().strip()
    uvicorn.run("saes_ai.server.app:app", host="0.0.0.0", port=8070, reload=True, log_level=log_level)


if __name__ == "__main__":
    main()
