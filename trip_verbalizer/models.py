"""
Pydantic models for the Trip Verbalizer pipeline.

This module defines all data models used throughout the pipeline,
including input schemas, intermediate representations, and output structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class EventType(str, Enum):
    """Types of driving events."""
    HARD_BRAKING = "hard_braking"
    HARD_BRAKE = "hard_brake"  # Alias
    SHARP_TURN = "sharp_turn"
    RAPID_ACCELERATION = "rapid_acceleration"
    ACCELERATION = "acceleration"  # Alias
    OVER_SPEED = "over_speed"
    IDLE = "idle"
    COLLISION = "collision"
    LANE_DEPARTURE = "lane_departure"
    TAILGATING = "tailgating"
    OTHER = "other"


class EventSeverity(str, Enum):
    """Severity levels for driving events."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class PhaseType(str, Enum):
    """Types of trip phases."""
    START = "start"
    ACCELERATION = "acceleration"
    CRUISING = "cruising"
    DECELERATION = "deceleration"
    STOP = "stop"
    IDLE = "idle"
    TURNING = "turning"


class TurnDirection(str, Enum):
    """Turn directions."""
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    U_TURN = "u_turn"


class VehicleConditionStatus(str, Enum):
    """Vehicle condition status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class GenerationMode(str, Enum):
    """Generation mode for trip verbalization."""
    NARRATIVE = "narrative"
    NAVIGATION_PAST = "navigation_past"


class NavigationAction(str, Enum):
    """Navigation action types for structured instructions."""
    START = "start"
    GO_STRAIGHT = "go_straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"
    U_TURN = "u_turn"
    ARRIVE = "arrive"


# =============================================================================
# Input Models - Raw Trip Data
# =============================================================================

class Coordinate(BaseModel):
    """Geographic coordinate with timestamp."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees", alias="lat")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees", alias="lng")
    timestamp: datetime = Field(..., description="UTC timestamp")
    altitude: float | None = Field(None, description="Altitude in meters")
    accuracy: float | None = Field(None, ge=0, description="GPS accuracy in meters")
    
    model_config = {"populate_by_name": True}
    
    @property
    def lat_lng(self) -> tuple[float, float]:
        """Return (lat, lng) tuple."""
        return (self.latitude, self.longitude)


class SpeedData(BaseModel):
    """Speed measurement at a point in time."""
    timestamp: datetime = Field(..., description="UTC timestamp")
    speed_kmh: float = Field(..., ge=0, description="Speed in km/h")
    speed_limit: float | None = Field(None, ge=0, description="Speed limit in km/h")
    
    @property
    def is_overspeeding(self) -> bool:
        """Check if speed exceeds limit."""
        if self.speed_limit is None:
            return False
        return self.speed_kmh > self.speed_limit


class DrivingEvent(BaseModel):
    """A driving event recorded during the trip."""
    event_id: str | None = Field(None, description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="When the event occurred")
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    severity: EventSeverity | None = Field(None, description="Event severity")
    value: float | None = Field(None, description="Event magnitude value")
    g_force: float | None = Field(None, description="G-force value")
    duration_seconds: float | None = Field(None, ge=0, description="Event duration")
    description: str | None = Field(None, description="Event description")
    location: dict | None = Field(None, description="Location object with lat/lng")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {"populate_by_name": True}
    
    @model_validator(mode="after")
    def extract_location(self) -> "DrivingEvent":
        """Extract lat/lng from location object if present."""
        if self.location and isinstance(self.location, dict):
            if self.latitude is None:
                self.latitude = self.location.get("lat")
            if self.longitude is None:
                self.longitude = self.location.get("lng")
        if self.event_id is None:
            self.event_id = f"evt_{self.timestamp.isoformat()}"
        return self


class VehicleCondition(BaseModel):
    """Vehicle condition data."""
    timestamp: datetime = Field(..., description="When condition was recorded")
    fuel_level: float | None = Field(None, ge=0, le=100, description="Fuel level percentage")
    battery_voltage: float | None = Field(None, ge=0, description="Battery voltage")
    engine_temp: float | None = Field(None, description="Engine temperature in Celsius")
    oil_pressure: float | None = Field(None, ge=0, description="Oil pressure")
    tire_pressure: dict[str, float] | None = Field(None, description="Tire pressures")
    odometer: float | None = Field(None, ge=0, description="Odometer reading in km")
    status: VehicleConditionStatus = Field(
        default=VehicleConditionStatus.NORMAL,
        description="Overall condition status"
    )
    warnings: list[str] = Field(default_factory=list, description="Active warnings")


class DriverInfo(BaseModel):
    """Driver information."""
    driver_id: str = Field(..., description="Unique driver identifier")
    name: str | None = Field(None, description="Driver name")
    license_number: str | None = Field(None, description="License number")
    rating: float | None = Field(None, ge=0, le=5, description="Driver rating")
    total_trips: int | None = Field(None, ge=0, description="Total trips completed")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TripData(BaseModel):
    """
    Complete trip data input model.
    
    This is the main input schema that contains all telematics data
    for a single vehicle trip.
    """
    trip_id: str = Field(..., description="Unique trip identifier")
    vehicle_id: str = Field(..., description="Vehicle identifier")
    
    # Route data
    route: list[Coordinate] = Field(..., min_length=2, description="Route coordinates")
    
    # Speed data
    speed_data: list[SpeedData] = Field(default_factory=list, description="Speed readings")
    
    # Events
    events: list[DrivingEvent] = Field(default_factory=list, description="Driving events")
    
    # Vehicle and driver - support both list and dict formats
    vehicle_conditions: list[VehicleCondition] | dict[str, Any] | None = Field(
        default=None,
        description="Vehicle condition readings"
    )
    driver: DriverInfo | None = Field(None, description="Driver information")
    driver_info: dict[str, Any] | None = Field(None, description="Driver info (alternate format)")
    
    # Timestamps
    start_time: datetime | None = Field(None, description="Trip start time")
    end_time: datetime | None = Field(None, description="Trip end time")
    
    # Additional fields from alternate formats
    start_location: dict[str, Any] | None = Field(None, description="Start location object")
    end_location: dict[str, Any] | None = Field(None, description="End location object")
    distance_km: float | None = Field(None, description="Trip distance in km")
    duration_seconds: float | None = Field(None, description="Trip duration in seconds")
    average_speed_kmh: float | None = Field(None, description="Average speed")
    max_speed_kmh: float | None = Field(None, description="Max speed")
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional trip metadata")
    
    model_config = {"populate_by_name": True, "extra": "allow"}
    
    @model_validator(mode="after")
    def validate_trip_times(self) -> "TripData":
        """Infer start/end times from route if not provided."""
        if self.route:
            if self.start_time is None:
                self.start_time = self.route[0].timestamp
            if self.end_time is None:
                self.end_time = self.route[-1].timestamp
        return self
    
    @property
    def duration_seconds(self) -> float | None:
        """Calculate trip duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# =============================================================================
# Enriched Models - After Geo Enrichment
# =============================================================================

class GeoLocation(BaseModel):
    """Enriched geographic location with reverse geocoding data."""
    latitude: float
    longitude: float
    timestamp: datetime
    
    # Geocoded data
    road_name: str | None = Field(None, description="Name of the road")
    locality: str | None = Field(None, description="Neighborhood or locality")
    city: str | None = Field(None, description="City name")
    state: str | None = Field(None, description="State or province")
    country: str | None = Field(None, description="Country name")
    postal_code: str | None = Field(None, description="Postal/ZIP code")
    
    # Computed fields
    display_name: str | None = Field(None, description="Full formatted address")
    
    @property
    def short_location(self) -> str:
        """Get short location description."""
        parts = [p for p in [self.road_name, self.locality, self.city] if p]
        return ", ".join(parts) if parts else f"{self.latitude:.4f}, {self.longitude:.4f}"


class EnrichedCoordinate(BaseModel):
    """Coordinate enriched with geocoding and spatial data."""
    # Original coordinate data
    latitude: float
    longitude: float
    timestamp: datetime
    altitude: float | None = None
    
    # Geocoded location
    location: GeoLocation | None = None
    
    # Spatial analysis
    bearing: float | None = Field(None, ge=0, lt=360, description="Bearing in degrees")
    direction: str | None = Field(None, description="Cardinal direction")
    distance_from_prev: float | None = Field(None, ge=0, description="Distance from previous point in meters")
    
    # Speed at this point
    speed_kmh: float | None = None
    
    @classmethod
    def from_coordinate(cls, coord: Coordinate, location: GeoLocation | None = None) -> "EnrichedCoordinate":
        """Create enriched coordinate from base coordinate."""
        return cls(
            latitude=coord.latitude,
            longitude=coord.longitude,
            timestamp=coord.timestamp,
            altitude=coord.altitude,
            location=location,
        )


# =============================================================================
# Navigation Instruction Models
# =============================================================================

class NavigationInstruction(BaseModel):
    """
    Structured navigation instruction for past-tense driving reports.
    
    Maps to Google Maps-style turn-by-turn directions.
    """
    action: NavigationAction = Field(..., description="Navigation action type")
    location: str | None = Field(None, description="Location name for start/arrive")
    distance_m: float | None = Field(None, ge=0, description="Distance in meters")
    road: str | None = Field(None, description="Road name")
    direction: str | None = Field(None, description="Cardinal direction (e.g., northwest)")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"action": self.action.value}
        if self.location:
            result["location"] = self.location
        if self.distance_m is not None:
            result["distance_m"] = round(self.distance_m)
        if self.road:
            result["road"] = self.road
        if self.direction:
            result["direction"] = self.direction
        return result


class NavigationSegment(BaseModel):
    """
    A segment of the route between direction changes.
    
    Aggregates consecutive points traveling in the same direction
    on the same road.
    """
    start_index: int = Field(..., description="Index in enriched_route")
    end_index: int = Field(..., description="Index in enriched_route")
    road_name: str | None = None
    direction: str | None = Field(None, description="Cardinal direction")
    distance_m: float = Field(0.0, ge=0, description="Segment distance in meters")
    movement_type: str = Field("straight", description="straight, slight_left, left, right, etc.")
    turn_angle: float | None = Field(None, description="Turn angle at end of segment")


# =============================================================================
# Analysis Models - Spatial and Temporal Analysis Results
# =============================================================================

class Turn(BaseModel):
    """Detected turn in the route."""
    timestamp: datetime
    latitude: float
    longitude: float
    direction: TurnDirection
    angle: float = Field(..., description="Turn angle in degrees")
    location_name: str | None = None
    severity: str = Field(default="normal", description="Turn severity")


class Phase(BaseModel):
    """A phase/segment of the trip."""
    phase_type: PhaseType
    start_time: datetime
    end_time: datetime
    start_location: GeoLocation | None = None
    end_location: GeoLocation | None = None
    
    # Statistics
    duration_seconds: float
    distance_meters: float | None = None
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    min_speed_kmh: float | None = None
    
    # Phase-specific data
    description: str | None = None


class SpeedAnomaly(BaseModel):
    """Detected speed anomaly."""
    timestamp: datetime
    latitude: float
    longitude: float
    speed_kmh: float
    expected_speed_kmh: float
    anomaly_type: str = Field(..., description="Type of anomaly (spike, sudden_drop, etc.)")
    severity: EventSeverity


class AnalyzedEvent(BaseModel):
    """Event after analysis with severity scoring."""
    # Original event data
    event_id: str
    event_type: EventType
    timestamp: datetime
    latitude: float
    longitude: float
    
    # Analysis results
    severity: EventSeverity
    severity_score: float = Field(..., ge=0, le=10, description="Numerical severity score")
    
    # Context
    location_name: str | None = None
    road_name: str | None = None
    speed_at_event: float | None = None
    
    # Description
    description: str
    impact_assessment: str | None = None


# =============================================================================
# Output Models - Semantic Summary and Narration
# =============================================================================

class DriverBehaviorSummary(BaseModel):
    """Summary of driver behavior during the trip."""
    overall_rating: float = Field(..., ge=0, le=10, description="Overall behavior score")
    
    # Event counts
    hard_braking_count: int = 0
    sharp_turn_count: int = 0
    rapid_acceleration_count: int = 0
    over_speed_count: int = 0
    
    # Behavior indicators
    aggressive_driving: bool = False
    smooth_driving: bool = True
    speed_compliance: float = Field(default=100.0, ge=0, le=100, description="% time within speed limit")
    
    # Narrative elements
    behavior_summary: str | None = None
    improvement_suggestions: list[str] = Field(default_factory=list)


class RouteDescription(BaseModel):
    """Description of the route taken."""
    start_location: str
    end_location: str
    total_distance_km: float
    
    # Key waypoints
    major_roads: list[str] = Field(default_factory=list)
    cities_passed: list[str] = Field(default_factory=list)
    notable_locations: list[str] = Field(default_factory=list)
    
    # Route characteristics
    route_type: str = Field(default="mixed", description="urban, highway, rural, mixed")
    general_direction: str | None = None


class TripSummary(BaseModel):
    """High-level trip summary."""
    trip_id: str
    duration: str  # Human-readable duration
    distance: str  # Human-readable distance
    start_time: str  # Formatted start time
    end_time: str  # Formatted end time
    start_location: str
    end_location: str
    average_speed: str
    max_speed: str
    event_count: int
    
    # Quick stats
    idle_time: str | None = None
    fuel_used: str | None = None


class SemanticSummary(BaseModel):
    """
    Complete semantic summary of the trip.
    
    This is the structured output that feeds into the LLM prompt builder.
    """
    trip_summary: TripSummary
    phases: list[Phase]
    events: list[AnalyzedEvent]
    route_description: RouteDescription
    driver_behavior: DriverBehaviorSummary
    
    # Additional context
    notable_observations: list[str] = Field(default_factory=list)
    anomalies: list[SpeedAnomaly] = Field(default_factory=list)
    
    # Raw data for LLM context
    enriched_route: list[EnrichedCoordinate] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)


class NarrationOutput(BaseModel):
    """Final output from the pipeline."""
    # Main narration
    narration: str = Field(..., description="Natural language trip narration")
    
    # Structured data
    semantic_summary: SemanticSummary
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str | None = None
    processing_time_seconds: float | None = None
    
    # Debug information
    debug_logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# Cache Models
# =============================================================================

class GeocodeCache(BaseModel):
    """Cached geocoding result."""
    latitude: float
    longitude: float
    result: GeoLocation
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="nominatim", description="Geocoding service used")
    
    @property
    def cache_key(self) -> str:
        """Generate cache key from coordinates."""
        return f"{self.latitude:.6f},{self.longitude:.6f}"
