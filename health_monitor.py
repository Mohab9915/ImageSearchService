"""Authenticated, coalesced functional checks; results never include credentials."""
import asyncio
import hmac
import os
import time
from datetime import datetime, timezone
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse


def install(app, checks):
    cached = None
    expires = 0
    task = None

    app.state.functional_failure_until = 0

    @app.middleware("http")
    async def track_errors(request, call_next):
        if request.url.path.startswith("/health"):
            return await call_next(request)
        try:
            response = await call_next(request)
        except Exception:
            app.state.functional_failure_until = time.monotonic() + 300
            raise
        if response.status_code >= 500:
            app.state.functional_failure_until = time.monotonic() + 300
        return response

    async def run():
        nonlocal cached, expires
        async def check(name, probe):
            try:
                await asyncio.wait_for(probe(), timeout=90)
                return name, True
            except Exception:
                return name, False
        results = dict(await asyncio.gather(*(check(n, p) for n, p in checks.items())))
        results["recent_requests"] = time.monotonic() >= app.state.functional_failure_until
        cached = {
            "status": "operational" if results and all(results.values()) else "degraded",
            "checks": results,
            "checkedAt": datetime.now(timezone.utc).isoformat(),
        }
        expires = time.monotonic() + (300 if cached["status"] == "operational" else 60)
        return cached

    @app.get("/health/functional")
    async def functional_health(x_status_token: str = Header(default="")):
        nonlocal task
        expected = os.getenv("STATUS_CHECK_TOKEN", "")
        if not expected or not hmac.compare_digest(x_status_token, expected):
            raise HTTPException(401, "Unauthorized")
        if time.monotonic() >= expires:
            if task is None or task.done():
                task = asyncio.create_task(run())
            await asyncio.shield(task)
        result = dict(cached)
        if time.monotonic() < app.state.functional_failure_until:
            result = {**cached, "status": "degraded", "checks": {**cached["checks"], "recent_requests": False}}
        return JSONResponse(result, status_code=200 if result["status"] == "operational" else 503,
                            headers={"Cache-Control": "no-store"})

