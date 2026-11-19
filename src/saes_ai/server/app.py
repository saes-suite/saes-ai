"""FastAPI standalone server for SAES-AI."""

import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from saes_ai.server.routes import router as chat_router

logger = logging.getLogger("saes_ai.server")

_ANSI_RESET = "\033[0m"
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[38;5;208m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class _ColorFormatter(logging.Formatter):
    """Console formatter that colorizes log lines by severity level."""

    def __init__(self, fmt: str, datefmt: str | None = None, *, use_color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_color:
            return formatted
        color = _LEVEL_COLORS.get(record.levelno)
        if not color:
            return formatted
        return f"{color}{formatted}{_ANSI_RESET}"


def _should_use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR") or os.getenv("CLICOLOR_FORCE"):
        return True
    return sys.stderr.isatty()


def configure_logging(level: str | None = None) -> None:
    """Configure structured console logging for the ``saes_ai`` package."""
    resolved = (level or os.getenv("SAES_AI_LOG_LEVEL") or "INFO").upper().strip()
    numeric_level = getattr(logging, resolved, logging.INFO)

    pkg_logger = logging.getLogger("saes_ai")
    pkg_logger.setLevel(numeric_level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            _ColorFormatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
                use_color=_should_use_color(),
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
