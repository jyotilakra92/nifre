"""Admin API routes for auto-tuning control."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from autotune.policy import TuningGoal


class TuningUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    goal: Optional[Literal["latency", "throughput", "balanced"]] = None


def tuning_unavailable() -> JSONResponse:
    return JSONResponse(
        {
            "available": False,
            "enabled": False,
            "goal": None,
            "message": "auto-tuning requires observability to be enabled",
        },
        status_code=503,
    )


def get_tuning_payload(request: Request) -> dict:
    controller = getattr(request.app.state, "tuning_controller", None)
    if controller is None:
        return {
            "available": False,
            "enabled": False,
            "goal": None,
        }
    return controller.snapshot()


def update_tuning(request: Request, body: TuningUpdateRequest) -> JSONResponse:
    controller = getattr(request.app.state, "tuning_controller", None)
    if controller is None:
        return tuning_unavailable()

    if body.goal is not None:
        controller.set_goal(TuningGoal(body.goal))
    if body.enabled is not None:
        controller.set_enabled(body.enabled)

    return JSONResponse(controller.snapshot())


def register_tuning_routes(app: FastAPI) -> None:
    @app.get("/v1/admin/tuning")
    def admin_get_tuning(request: Request):
        controller = getattr(request.app.state, "tuning_controller", None)
        if controller is None:
            return tuning_unavailable()
        return JSONResponse(controller.snapshot())

    @app.post("/v1/admin/tuning")
    def admin_update_tuning(body: TuningUpdateRequest, request: Request):
        return update_tuning(request, body)

    @app.get("/observability/api/tuning", include_in_schema=False)
    def observability_tuning(request: Request):
        controller = getattr(request.app.state, "tuning_controller", None)
        if controller is None:
            return JSONResponse(get_tuning_payload(request))
        return JSONResponse(controller.snapshot())
