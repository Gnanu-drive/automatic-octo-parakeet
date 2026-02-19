"""
Spatial Analysis Module

This module handles spatial computations including:
- Bearing and heading calculations
- Turn detection (left/right/U-turn)
- Direction change analysis
- Route geometry analysis
"""

import logging
import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point

from .models import (
    Coordinate,
    EnrichedCoordinate,
    GeoLocation,
    Turn,
    TurnDirection,
)
from .utils.helpers import (
    bearing_to_cardinal,
    bearing_to_direction_phrase,
    calculate_bearing,
    haversine_distance,
    normalize_angle,
)


logger = logging.getLogger(__name__)


class SpatialAnalyzer:
    """
    Performs spatial analysis on trip route data.
    
    Computes bearings, directions, distances, and detects turns.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize spatial analyzer.
        
        Args:
            config: Configuration dictionary with spatial settings
        """
        self.config = config or {}
        spatial_config = self.config.get("spatial", {})
        
        # Turn detection thresholds (degrees)
        self.turn_threshold_minor = spatial_config.get("turn_threshold_minor", 20)
        self.turn_threshold_moderate = spatial_config.get("turn_threshold_moderate", 45)
        self.turn_threshold_sharp = spatial_config.get("turn_threshold_sharp", 90)
        
        # Cardinal direction precision
        self.cardinal_precision = spatial_config.get("cardinal_directions", 8)
    
    def compute_bearings(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> list[EnrichedCoordinate]:
        """
        Compute bearing and direction for each coordinate.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            Same list with bearing/direction fields populated
        """
        if len(coordinates) < 2:
            return coordinates
        
        for i in range(len(coordinates) - 1):
            curr = coordinates[i]
            next_coord = coordinates[i + 1]
            
            # Calculate bearing to next point
            bearing = calculate_bearing(
                curr.latitude, curr.longitude,
                next_coord.latitude, next_coord.longitude
            )
            
            curr.bearing = bearing
            curr.direction = bearing_to_cardinal(bearing, self.cardinal_precision)
            
            # Calculate distance from previous point
            if i > 0:
                prev = coordinates[i - 1]
                distance = haversine_distance(
                    prev.latitude, prev.longitude,
                    curr.latitude, curr.longitude
                )
                curr.distance_from_prev = distance
        
        # Handle last point
        if len(coordinates) >= 2:
            last = coordinates[-1]
            second_last = coordinates[-2]
            last.bearing = second_last.bearing
            last.direction = second_last.direction
            last.distance_from_prev = haversine_distance(
                second_last.latitude, second_last.longitude,
                last.latitude, last.longitude
            )
        
        return coordinates
    
    def detect_turns(
        self,
        coordinates: list[EnrichedCoordinate],
        min_angle: float = 20.0,
    ) -> list[Turn]:
        """
        Detect turns in the route based on bearing changes.
        
        Args:
            coordinates: List of enriched coordinates with bearings
            min_angle: Minimum angle to consider as a turn
            
        Returns:
            List of detected turns
        """
        turns: list[Turn] = []
        
        if len(coordinates) < 3:
            return turns
        
        # Compute heading changes
        heading_changes = self._compute_heading_changes(coordinates)
        
        # Detect significant heading changes
        for i, change in enumerate(heading_changes):
            if abs(change) >= min_angle:
                coord = coordinates[i + 1]  # Turn occurs at middle point
                
                # Determine turn direction
                direction = self._classify_turn_direction(change)
                
                # Determine severity
                severity = self._classify_turn_severity(abs(change))
                
                # Get location name
                location_name = None
                if coord.location:
                    location_name = coord.location.short_location
                
                turn = Turn(
                    timestamp=coord.timestamp,
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    direction=direction,
                    angle=abs(change),
                    location_name=location_name,
                    severity=severity,
                )
                turns.append(turn)
        
        logger.info(f"Detected {len(turns)} turns in route")
        return turns
    
    def _compute_heading_changes(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> list[float]:
        """
        Compute heading change at each point.
        
        Returns list of heading changes (positive = right, negative = left).
        """
        changes: list[float] = []
        
        for i in range(len(coordinates) - 2):
            curr = coordinates[i]
            next_coord = coordinates[i + 1]
            
            if curr.bearing is not None and next_coord.bearing is not None:
                # Calculate angle difference
                diff = next_coord.bearing - curr.bearing
                
                # Normalize to -180 to 180
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                
                changes.append(diff)
            else:
                changes.append(0.0)
        
        return changes
    
    def _classify_turn_direction(self, angle_change: float) -> TurnDirection:
        """Classify turn direction based on angle change."""
        if abs(angle_change) >= 150:
            return TurnDirection.U_TURN
        elif angle_change > 0:
            return TurnDirection.RIGHT
        elif angle_change < 0:
            return TurnDirection.LEFT
        else:
            return TurnDirection.STRAIGHT
    
    def _classify_turn_severity(self, angle: float) -> str:
        """Classify turn severity based on angle magnitude."""
        if angle >= self.turn_threshold_sharp:
            return "sharp"
        elif angle >= self.turn_threshold_moderate:
            return "moderate"
        elif angle >= self.turn_threshold_minor:
            return "minor"
        else:
            return "slight"
    
    def compute_total_distance(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> float:
        """
        Compute total distance of the route.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            Total distance in meters
        """
        total = 0.0
        
        for coord in coordinates:
            if coord.distance_from_prev:
                total += coord.distance_from_prev
        
        return total
    
    def compute_general_direction(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> str:
        """
        Compute the general direction of travel from start to end.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            General direction phrase (e.g., "northwest")
        """
        if len(coordinates) < 2:
            return "unknown"
        
        start = coordinates[0]
        end = coordinates[-1]
        
        bearing = calculate_bearing(
            start.latitude, start.longitude,
            end.latitude, end.longitude
        )
        
        return bearing_to_direction_phrase(bearing)
    
    def get_major_roads(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> list[str]:
        """
        Extract unique major roads from the route.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            List of unique road names
        """
        roads: list[str] = []
        seen: set[str] = set()
        
        for coord in coordinates:
            if coord.location and coord.location.road_name:
                road = coord.location.road_name
                if road not in seen:
                    roads.append(road)
                    seen.add(road)
        
        return roads
    
    def get_cities_passed(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> list[str]:
        """
        Extract unique cities passed through.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            List of unique city names
        """
        cities: list[str] = []
        seen: set[str] = set()
        
        for coord in coordinates:
            if coord.location and coord.location.city:
                city = coord.location.city
                if city not in seen:
                    cities.append(city)
                    seen.add(city)
        
        return cities
    
    def analyze_route_geometry(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> dict[str, Any]:
        """
        Analyze overall route geometry using shapely.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            Dictionary with geometry metrics
        """
        if len(coordinates) < 2:
            return {"error": "Insufficient coordinates"}
        
        # Create LineString from coordinates
        points = [(c.longitude, c.latitude) for c in coordinates]
        line = LineString(points)
        
        # Compute metrics
        start_point = Point(coordinates[0].longitude, coordinates[0].latitude)
        end_point = Point(coordinates[-1].longitude, coordinates[-1].latitude)
        
        # Approximate bounding box (degrees)
        bounds = line.bounds  # (minx, miny, maxx, maxy)
        
        # Route straightness (ratio of direct distance to actual distance)
        direct_distance = haversine_distance(
            coordinates[0].latitude, coordinates[0].longitude,
            coordinates[-1].latitude, coordinates[-1].longitude
        )
        actual_distance = self.compute_total_distance(coordinates)
        
        straightness = direct_distance / actual_distance if actual_distance > 0 else 1.0
        
        return {
            "bounds": {
                "min_lon": bounds[0],
                "min_lat": bounds[1],
                "max_lon": bounds[2],
                "max_lat": bounds[3],
            },
            "direct_distance_m": direct_distance,
            "actual_distance_m": actual_distance,
            "straightness_ratio": straightness,
            "is_straight": straightness > 0.9,
            "is_circular": self._is_circular_route(coordinates),
        }
    
    def _is_circular_route(
        self,
        coordinates: list[EnrichedCoordinate],
        threshold_m: float = 500.0
    ) -> bool:
        """Check if route starts and ends at approximately the same location."""
        if len(coordinates) < 2:
            return False
        
        start = coordinates[0]
        end = coordinates[-1]
        
        distance = haversine_distance(
            start.latitude, start.longitude,
            end.latitude, end.longitude
        )
        
        return distance <= threshold_m
    
    def describe_route_direction(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> str:
        """
        Generate a human-readable description of the route direction.
        
        Args:
            coordinates: List of enriched coordinates
            
        Returns:
            Direction description string
        """
        if len(coordinates) < 2:
            return "stationary"
        
        general_dir = self.compute_general_direction(coordinates)
        geometry = self.analyze_route_geometry(coordinates)
        
        if geometry.get("is_circular"):
            return "circular route returning to the starting point"
        elif geometry.get("is_straight", False):
            return f"straight route heading {general_dir}"
        else:
            return f"winding route generally heading {general_dir}"
    
    def segment_by_direction_changes(
        self,
        coordinates: list[EnrichedCoordinate],
        threshold: float = 45.0
    ) -> list[list[EnrichedCoordinate]]:
        """
        Segment the route by significant direction changes.
        
        Args:
            coordinates: List of enriched coordinates
            threshold: Minimum angle to trigger new segment
            
        Returns:
            List of route segments
        """
        if len(coordinates) < 2:
            return [coordinates]
        
        segments: list[list[EnrichedCoordinate]] = []
        current_segment: list[EnrichedCoordinate] = [coordinates[0]]
        
        heading_changes = self._compute_heading_changes(coordinates)
        
        for i, change in enumerate(heading_changes):
            current_segment.append(coordinates[i + 1])
            
            if abs(change) >= threshold:
                # Start new segment
                segments.append(current_segment)
                current_segment = [coordinates[i + 1]]
        
        # Add remaining segment
        if i + 2 < len(coordinates):
            current_segment.extend(coordinates[i + 2:])
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
