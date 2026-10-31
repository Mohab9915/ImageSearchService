import asyncio
import os
import unittest
from unittest.mock import patch
import httpx
from fastapi import FastAPI
from health_monitor import install


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_authentication_cache_concurrency_and_recent_failure(self):
        calls = []
        async def probe():
            calls.append(1)
            await asyncio.sleep(0.01)
        app = FastAPI()
        install(app, {"ai": probe})
        @app.get("/broken")
        async def broken():
            from fastapi.responses import Response
            return Response(status_code=500)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch.dict(os.environ, {"STATUS_CHECK_TOKEN": "private"}):
                self.assertEqual((await client.get("/health/functional")).status_code, 401)
                self.assertEqual(calls, [])
                headers = {"x-status-token": "private"}
                responses = await asyncio.gather(*(client.get("/health/functional", headers=headers) for _ in range(3)))
                self.assertTrue(all(r.status_code == 200 for r in responses))
                self.assertEqual(len(calls), 1)
                await client.get("/broken")
                response = await client.get("/health/functional", headers=headers)
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["checks"]["recent_requests"])

    async def test_probe_failure_is_degraded_without_leaking_errors(self):
        async def broken():
            raise RuntimeError("secret credentials")
        app = FastAPI()
        install(app, {"database": broken})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch.dict(os.environ, {"STATUS_CHECK_TOKEN": "private"}):
                response = await client.get("/health/functional", headers={"x-status-token": "private"})
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["checks"]["database"])
                self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
