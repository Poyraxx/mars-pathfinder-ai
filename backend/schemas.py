"""Explicit Mars contracts. No implicit WGS84 / GeoJSON coordinates."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MarsPoint(Contract):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, lt=180)
    body: Literal[499] = 499
    latitude_type: Literal["planetocentric"] = "planetocentric"


class Profile(Contract):
    mode: Literal["human_eva_research"] = "human_eva_research"
    kind: Literal["astronaut", "crew_cart", "rover"] = "astronaut"
    max_slope_deg: float = Field(default=20, ge=1, le=45)
    speed_m_s: float = Field(default=0.5, ge=0.01, le=2)
    clearance_m: float = Field(default=1.0, ge=0, le=5)
    obstacle_step_m: float = Field(default=0.15, ge=0.05, le=1.0)


class MissionRequest(Contract):
    region_id: str = "gale"
    start: MarsPoint
    goal: MarsPoint
    stops: list[MarsPoint] = Field(default_factory=list, max_length=5)
    profile: Profile = Field(default_factory=Profile)
    avoid_features: list[str] = Field(default_factory=list, max_length=10)
    eva: "EvaPlan | None" = None


class EvaPlan(Contract):
    budget_min: float = Field(default=420, ge=10, le=10080)
    work_min: list[float] = Field(default_factory=list, max_length=5)
    date: str | None = Field(default=None, max_length=32)


class ObservationRequest(Contract):
    frame_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.-]+$")
    base_map_version: int = Field(ge=0)


class PlanRequest(Contract):
    expected_map_version: int | None = Field(default=None, ge=0)

MissionRequest.model_rebuild()
