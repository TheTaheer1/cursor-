"""
app.py
FastAPI server for EvoAI Lab. Exposes REST + WebSocket APIs and serves
the React frontend's production build.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.env.evoai_env import EvoAIEnv


app = FastAPI(title="EvoAI Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_env: EvoAIEnv | None = None
_env_error: str | None = None


def get_env() -> EvoAIEnv:
    global _env, _env_error
    if _env is not None:
        return _env
    try:
        _env = EvoAIEnv()
        _env_error = None
        return _env
    except Exception as e:
        _env_error = str(e)
        raise HTTPException(status_code=503, detail=f"Environment unavailable: {e}")


class RunStepsBody(BaseModel):
    n: int = 1


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    return {"status": "ok", "groq_key_set": has_key, "env_error": _env_error}


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    env = get_env()
    return env.state()


@app.get("/api/reward-curve")
async def get_reward_curve() -> Any:
    env = get_env()
    return env.pipeline.dataset_builder.get_reward_curve()


@app.get("/api/failures")
async def get_failures() -> Any:
    env = get_env()
    return env.pipeline.dataset_builder.get_recent_failures(20)


@app.get("/api/calibration-map")
async def get_calibration_map() -> Any:
    env = get_env()
    return env.pipeline.calibration_map.to_dict()


@app.post("/api/reset")
async def reset_env() -> Dict[str, Any]:
    env = get_env()
    return env.reset()


@app.post("/api/run-steps")
async def run_steps(body: RunStepsBody) -> Dict[str, Any]:
    env = get_env()
    n = max(1, min(int(body.n or 1), 100))
    last: Dict[str, Any] = {}
    for _ in range(n):
        last = await env.step()
    return {"ran": n, "state": env.state(), "last": last}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        try:
            env = get_env()
        except HTTPException as e:
            await websocket.send_text(json.dumps({"error": e.detail}))
            await websocket.close()
            return

        await websocket.send_text(json.dumps({"type": "state", **env.state()}))

        while True:
            try:
                # Honour client-driven pacing if the client sends anything.
                # Otherwise step continuously with a small pause.
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                    try:
                        client_msg = json.loads(msg)
                        if client_msg.get("action") == "stop":
                            break
                    except Exception:
                        pass
                except asyncio.TimeoutError:
                    pass

                step_result = await env.step()
                payload = {
                    "type": "step",
                    "step": step_result["observation"]["step"],
                    "reward": step_result["reward"],
                    "calibration_map": step_result["observation"]["calibration_map"],
                    "zone_counts": {
                        "zone_c": step_result["observation"]["zone_c_count"],
                        "zone_b": step_result["observation"]["zone_b_count"],
                        "green": step_result["observation"]["green_count"],
                    },
                    "failure": step_result["info"].get("failure"),
                    "probe": step_result["info"].get("probe"),
                    "question": step_result["info"].get("question"),
                }
                await websocket.send_text(json.dumps(payload, default=str))
                await asyncio.sleep(0.5)
            except WebSocketDisconnect:
                break
            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
                except Exception:
                    break
                await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


# Mount the frontend build at the root if present.
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    async def root_index() -> JSONResponse:
        return JSONResponse(
            {
                "name": "EvoAI Lab",
                "tagline": (
                    "Most AI training only asks if the model is right or wrong. "
                    "We train for something harder: is it right about being right?"
                ),
                "frontend_built": False,
                "endpoints": [
                    "/api/health",
                    "/api/state",
                    "/api/reward-curve",
                    "/api/failures",
                    "/api/calibration-map",
                    "/api/reset",
                    "/api/run-steps",
                    "/ws/live",
                ],
            }
        )
