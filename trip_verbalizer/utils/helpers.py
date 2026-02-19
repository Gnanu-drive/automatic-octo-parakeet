"""
Helper functions for the Trip Verbalizer pipeline.
"""

import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config.yaml in the package directory
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str | None = None
) -> structlog.BoundLogger:
    """
    Set up structured logging for the pipeline.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        log_format: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure standard logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger("trip_verbalizer")


def format_time(dt: datetime, format_str: str = "%I:%M %p") -> str:
    """
    Format datetime to human-readable time string.
    
    Args:
        dt: Datetime object
        format_str: strftime format string
        
    Returns:
        Formatted time string (e.g., "9:00 AM")
    """
    return dt.strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration (e.g., "1 hour 30 minutes")
    """
    if seconds < 60:
        return f"{int(seconds)} seconds"
    
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if td.days > 0:
        parts.append(f"{td.days} day{'s' if td.days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if not parts:
        parts.append(f"{secs} seconds")
    
    return " ".join(parts)


def format_speed(speed_kmh: float, unit: str = "km/h") -> str:
    """
    Format speed value to human-readable string.
    
    Args:
        speed_kmh: Speed in km/h
        unit: Unit to display
        
    Returns:
        Formatted speed string (e.g., "60 km/h")
    """
    return f"{speed_kmh:.0f} {unit}"


def format_distance(distance_m: float) -> str:
    """
    Format distance to human-readable string with appropriate unit.
    
    Args:
        distance_m: Distance in meters
        
    Returns:
        Formatted distance (e.g., "2.5 km" or "500 m")
    """
    if distance_m >= 1000:
        return f"{distance_m / 1000:.1f} km"
    return f"{distance_m:.0f} m"


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in meters
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_phi / 2) ** 2 +
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to be within 0-360 degrees.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle (0-360)
    """
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle


def calculate_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate initial bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2) -
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    
    theta = math.atan2(x, y)
    bearing = math.degrees(theta)
    
    return normalize_angle(bearing)


def bearing_to_cardinal(bearing: float, precision: int = 8) -> str:
    """
    Convert bearing to cardinal direction.
    
    Args:
        bearing: Bearing in degrees (0-360)
        precision: Number of cardinal points (4, 8, or 16)
        
    Returns:
        Cardinal direction string (e.g., "N", "NE", "NNE")
    """
    directions_4 = ["N", "E", "S", "W"]
    directions_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    directions_16 = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    
    if precision == 4:
        directions = directions_4
    elif precision == 8:
        directions = directions_8
    else:
        directions = directions_16
    
    sector_size = 360 / len(directions)
    index = round(bearing / sector_size) % len(directions)
    
    return directions[index]


def bearing_to_direction_phrase(bearing: float) -> str:
    """
    Convert bearing to human-readable direction phrase.
    
    Args:
        bearing: Bearing in degrees (0-360)
        
    Returns:
        Direction phrase (e.g., "north", "northeast", "north-northwest")
    """
    direction_phrases = {
        "N": "north",
        "NNE": "north-northeast",
        "NE": "northeast",
        "ENE": "east-northeast",
        "E": "east",
        "ESE": "east-southeast",
        "SE": "southeast",
        "SSE": "south-southeast",
        "S": "south",
        "SSW": "south-southwest",
        "SW": "southwest",
        "WSW": "west-southwest",
        "W": "west",
        "WNW": "west-northwest",
        "NW": "northwest",
        "NNW": "north-northwest",
    }
    
    cardinal = bearing_to_cardinal(bearing, precision=16)
    return direction_phrases.get(cardinal, cardinal.lower())
