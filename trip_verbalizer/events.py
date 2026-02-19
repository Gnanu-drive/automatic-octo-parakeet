"""
Event Analysis Module

This module handles driving event analysis including:
- Hard braking detection and classification
- Sharp turn analysis
- Rapid acceleration detection
- Severity scoring for all events
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np

from .models import (
    AnalyzedEvent,
    Coordinate,
    DrivingEvent,
    EnrichedCoordinate,
    EventSeverity,
    EventType,
    GeoLocation,
    SpeedData,
)


logger = logging.getLogger(__name__)


class EventAnalyzer:
    """
    Analyzes driving events and assigns severity scores.
    
    Provides interpretation and context for each event.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize event analyzer.
        
        Args:
            config: Configuration dictionary with event settings
        """
        self.config = config or {}
        events_config = self.config.get("events", {})
        
        # Hard braking thresholds (g-force)
        braking_config = events_config.get("hard_braking", {})
        self.braking_mild = braking_config.get("mild", 0.3)
        self.braking_moderate = braking_config.get("moderate", 0.5)
        self.braking_severe = braking_config.get("severe", 0.7)
        
        # Sharp turn thresholds (degrees per second)
        turn_config = events_config.get("sharp_turn", {})
        self.turn_mild = turn_config.get("mild", 15)
        self.turn_moderate = turn_config.get("moderate", 30)
        self.turn_severe = turn_config.get("severe", 45)
        
        # Rapid acceleration thresholds (g-force)
        accel_config = events_config.get("rapid_acceleration", {})
        self.accel_mild = accel_config.get("mild", 0.3)
        self.accel_moderate = accel_config.get("moderate", 0.5)
        self.accel_severe = accel_config.get("severe", 0.7)
        
        # Severity weights
        weights = events_config.get("severity_weights", {})
        self.weight_braking = weights.get("braking", 1.2)
        self.weight_turn = weights.get("turn", 1.0)
        self.weight_acceleration = weights.get("acceleration", 0.8)
    
    def analyze_events(
        self,
        events: list[DrivingEvent],
        enriched_route: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> list[AnalyzedEvent]:
        """
        Analyze all driving events and assign severity scores.
        
        Args:
            events: List of raw driving events
            enriched_route: Enriched route coordinates (for context)
            speed_data: Optional speed data
            
        Returns:
            List of analyzed events with severity scores
        """
        analyzed: list[AnalyzedEvent] = []
        
        for event in events:
            analyzed_event = self._analyze_single_event(
                event, enriched_route, speed_data
            )
            analyzed.append(analyzed_event)
        
        # Sort by timestamp
        analyzed.sort(key=lambda e: e.timestamp)
        
        logger.info(f"Analyzed {len(analyzed)} events")
        return analyzed
    
    def _analyze_single_event(
        self,
        event: DrivingEvent,
        enriched_route: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> AnalyzedEvent:
        """Analyze a single event."""
        # Get location context
        location = self._find_nearest_location(
            event.latitude, event.longitude, enriched_route
        )
        
        # Get speed at event
        speed_at_event = self._get_speed_at_time(event.timestamp, speed_data)
        
        # Compute severity
        severity = self._compute_severity(event)
        severity_score = self._compute_severity_score(event)
        
        # Generate description
        description = self._generate_event_description(event, location, speed_at_event)
        
        # Generate impact assessment
        impact = self._assess_impact(event, severity)
        
        return AnalyzedEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            latitude=event.latitude,
            longitude=event.longitude,
            severity=severity,
            severity_score=severity_score,
            location_name=location.short_location if location else None,
            road_name=location.road_name if location else None,
            speed_at_event=speed_at_event,
            description=description,
            impact_assessment=impact,
        )
    
    def _find_nearest_location(
        self,
        lat: float,
        lon: float,
        enriched_route: list[EnrichedCoordinate]
    ) -> GeoLocation | None:
        """Find the nearest enriched coordinate for location context."""
        if not enriched_route:
            return None
        
        from .utils.helpers import haversine_distance
        
        min_distance = float('inf')
        nearest_location: GeoLocation | None = None
        
        for coord in enriched_route:
            if coord.location:
                distance = haversine_distance(lat, lon, coord.latitude, coord.longitude)
                if distance < min_distance:
                    min_distance = distance
                    nearest_location = coord.location
        
        return nearest_location
    
    def _get_speed_at_time(
        self,
        timestamp: datetime,
        speed_data: list[SpeedData] | None
    ) -> float | None:
        """Get speed at or near a specific timestamp."""
        if not speed_data:
            return None
        
        # Find closest speed reading
        min_diff = float('inf')
        closest_speed: float | None = None
        
        for reading in speed_data:
            diff = abs((reading.timestamp - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_speed = reading.speed_kmh
        
        # Only use if within 30 seconds
        if min_diff <= 30:
            return closest_speed
        return None
    
    def _compute_severity(self, event: DrivingEvent) -> EventSeverity:
        """Compute severity level for an event."""
        # Use existing severity if provided and valid
        if event.severity:
            return event.severity
        
        # Compute based on event type and value
        value = event.value or 0.0
        
        if event.event_type == EventType.HARD_BRAKING:
            if value >= self.braking_severe:
                return EventSeverity.CRITICAL
            elif value >= self.braking_moderate:
                return EventSeverity.SEVERE
            elif value >= self.braking_mild:
                return EventSeverity.MODERATE
            else:
                return EventSeverity.MINOR
        
        elif event.event_type == EventType.SHARP_TURN:
            if value >= self.turn_severe:
                return EventSeverity.SEVERE
            elif value >= self.turn_moderate:
                return EventSeverity.MODERATE
            elif value >= self.turn_mild:
                return EventSeverity.MINOR
            else:
                return EventSeverity.MINOR
        
        elif event.event_type == EventType.RAPID_ACCELERATION:
            if value >= self.accel_severe:
                return EventSeverity.SEVERE
            elif value >= self.accel_moderate:
                return EventSeverity.MODERATE
            elif value >= self.accel_mild:
                return EventSeverity.MINOR
            else:
                return EventSeverity.MINOR
        
        elif event.event_type == EventType.COLLISION:
            return EventSeverity.CRITICAL
        
        elif event.event_type == EventType.OVER_SPEED:
            # Severity based on how much over limit
            if value >= 30:  # 30+ km/h over
                return EventSeverity.CRITICAL
            elif value >= 20:
                return EventSeverity.SEVERE
            elif value >= 10:
                return EventSeverity.MODERATE
            else:
                return EventSeverity.MINOR
        
        # Default
        return EventSeverity.MINOR
    
    def _compute_severity_score(self, event: DrivingEvent) -> float:
        """Compute numerical severity score (0-10)."""
        value = event.value or 0.0
        
        # Base score from event type
        base_scores = {
            EventType.COLLISION: 10.0,
            EventType.HARD_BRAKING: 6.0,
            EventType.SHARP_TURN: 5.0,
            EventType.RAPID_ACCELERATION: 4.0,
            EventType.OVER_SPEED: 5.0,
            EventType.LANE_DEPARTURE: 4.0,
            EventType.TAILGATING: 5.0,
            EventType.IDLE: 1.0,
            EventType.OTHER: 3.0,
        }
        
        base_score = base_scores.get(event.event_type, 3.0)
        
        # Apply weight based on type
        weight = 1.0
        if event.event_type == EventType.HARD_BRAKING:
            weight = self.weight_braking
        elif event.event_type == EventType.SHARP_TURN:
            weight = self.weight_turn
        elif event.event_type == EventType.RAPID_ACCELERATION:
            weight = self.weight_acceleration
        
        # Modify based on value magnitude
        if value > 0:
            value_factor = min(2.0, 1.0 + value / 10.0)
        else:
            value_factor = 1.0
        
        score = min(10.0, base_score * weight * value_factor)
        return round(score, 2)
    
    def _generate_event_description(
        self,
        event: DrivingEvent,
        location: GeoLocation | None,
        speed: float | None
    ) -> str:
        """Generate human-readable event description."""
        location_str = ""
        if location:
            if location.road_name:
                location_str = f" on {location.road_name}"
            elif location.locality:
                location_str = f" near {location.locality}"
            elif location.city:
                location_str = f" in {location.city}"
        
        speed_str = f" at {speed:.0f} km/h" if speed else ""
        
        descriptions = {
            EventType.HARD_BRAKING: f"Hard braking event{location_str}{speed_str}",
            EventType.SHARP_TURN: f"Sharp turn{location_str}{speed_str}",
            EventType.RAPID_ACCELERATION: f"Rapid acceleration{location_str}{speed_str}",
            EventType.OVER_SPEED: f"Speeding detected{location_str}{speed_str}",
            EventType.COLLISION: f"Collision detected{location_str}",
            EventType.LANE_DEPARTURE: f"Lane departure{location_str}{speed_str}",
            EventType.TAILGATING: f"Tailgating detected{location_str}{speed_str}",
            EventType.IDLE: f"Extended idle period{location_str}",
        }
        
        base_desc = descriptions.get(
            event.event_type,
            f"{event.event_type.value.replace('_', ' ').title()}{location_str}"
        )
        
        # Add value context if available
        if event.value:
            if event.event_type == EventType.HARD_BRAKING:
                base_desc += f" ({event.value:.2f}g deceleration)"
            elif event.event_type == EventType.SHARP_TURN:
                base_desc += f" ({event.value:.0f}° angle)"
            elif event.event_type == EventType.RAPID_ACCELERATION:
                base_desc += f" ({event.value:.2f}g acceleration)"
            elif event.event_type == EventType.OVER_SPEED:
                base_desc += f" ({event.value:.0f} km/h over limit)"
        
        return base_desc
    
    def _assess_impact(
        self,
        event: DrivingEvent,
        severity: EventSeverity
    ) -> str:
        """Generate impact assessment for the event."""
        impacts = {
            EventSeverity.MINOR: "Low impact - within normal driving patterns",
            EventSeverity.MODERATE: "Moderate impact - indicates some aggressive driving",
            EventSeverity.SEVERE: "High impact - significant safety concern",
            EventSeverity.CRITICAL: "Critical impact - immediate safety risk",
        }
        
        return impacts.get(severity, "Unknown impact level")
    
    def compute_behavior_summary(
        self,
        analyzed_events: list[AnalyzedEvent]
    ) -> dict[str, Any]:
        """
        Compute overall driver behavior summary from events.
        
        Args:
            analyzed_events: List of analyzed events
            
        Returns:
            Dictionary with behavior metrics
        """
        if not analyzed_events:
            return {
                "overall_score": 10.0,
                "hard_braking_count": 0,
                "sharp_turn_count": 0,
                "rapid_acceleration_count": 0,
                "over_speed_count": 0,
                "aggressive_driving": False,
                "smooth_driving": True,
                "average_severity": 0.0,
            }
        
        # Count by type
        hard_braking = sum(
            1 for e in analyzed_events
            if e.event_type == EventType.HARD_BRAKING
        )
        sharp_turns = sum(
            1 for e in analyzed_events
            if e.event_type == EventType.SHARP_TURN
        )
        rapid_accel = sum(
            1 for e in analyzed_events
            if e.event_type == EventType.RAPID_ACCELERATION
        )
        over_speed = sum(
            1 for e in analyzed_events
            if e.event_type == EventType.OVER_SPEED
        )
        
        # Compute average severity
        severity_scores = [e.severity_score for e in analyzed_events]
        avg_severity = np.mean(severity_scores) if severity_scores else 0.0
        
        # Compute overall score (10 - penalties)
        total_events = len(analyzed_events)
        high_severity_count = sum(
            1 for e in analyzed_events
            if e.severity in [EventSeverity.SEVERE, EventSeverity.CRITICAL]
        )
        
        penalty = (total_events * 0.3) + (high_severity_count * 0.5)
        overall_score = max(0.0, 10.0 - penalty)
        
        # Determine driving style
        aggressive = (
            hard_braking >= 3 or
            high_severity_count >= 2 or
            avg_severity >= 6.0
        )
        smooth = total_events <= 2 and avg_severity <= 4.0
        
        return {
            "overall_score": round(overall_score, 2),
            "hard_braking_count": hard_braking,
            "sharp_turn_count": sharp_turns,
            "rapid_acceleration_count": rapid_accel,
            "over_speed_count": over_speed,
            "aggressive_driving": aggressive,
            "smooth_driving": smooth,
            "average_severity": round(avg_severity, 2),
            "total_events": total_events,
            "high_severity_events": high_severity_count,
        }
    
    def generate_behavior_narrative(
        self,
        behavior_summary: dict[str, Any]
    ) -> str:
        """
        Generate narrative description of driver behavior.
        
        Args:
            behavior_summary: Behavior summary from compute_behavior_summary
            
        Returns:
            Narrative description string
        """
        parts: list[str] = []
        
        score = behavior_summary.get("overall_score", 10.0)
        
        # Overall assessment
        if score >= 9.0:
            parts.append("The driver demonstrated excellent driving behavior")
        elif score >= 7.0:
            parts.append("The driver showed generally good driving behavior")
        elif score >= 5.0:
            parts.append("The driver exhibited moderate driving behavior with some concerns")
        else:
            parts.append("The driver showed concerning driving behavior")
        
        # Specific observations
        observations: list[str] = []
        
        hard_braking = behavior_summary.get("hard_braking_count", 0)
        if hard_braking > 0:
            observations.append(
                f"{hard_braking} hard braking event{'s' if hard_braking > 1 else ''}"
            )
        
        sharp_turns = behavior_summary.get("sharp_turn_count", 0)
        if sharp_turns > 0:
            observations.append(
                f"{sharp_turns} sharp turn{'s' if sharp_turns > 1 else ''}"
            )
        
        rapid_accel = behavior_summary.get("rapid_acceleration_count", 0)
        if rapid_accel > 0:
            observations.append(
                f"{rapid_accel} rapid acceleration event{'s' if rapid_accel > 1 else ''}"
            )
        
        over_speed = behavior_summary.get("over_speed_count", 0)
        if over_speed > 0:
            observations.append(
                f"{over_speed} speeding instance{'s' if over_speed > 1 else ''}"
            )
        
        if observations:
            parts.append(f"The trip included {', '.join(observations)}")
        else:
            parts.append("No significant driving events were recorded")
        
        # Style assessment
        if behavior_summary.get("aggressive_driving"):
            parts.append("Overall driving style was aggressive")
        elif behavior_summary.get("smooth_driving"):
            parts.append("Overall driving style was smooth and controlled")
        
        return ". ".join(parts) + "."
    
    def get_improvement_suggestions(
        self,
        behavior_summary: dict[str, Any]
    ) -> list[str]:
        """
        Generate improvement suggestions based on behavior.
        
        Args:
            behavior_summary: Behavior summary
            
        Returns:
            List of improvement suggestions
        """
        suggestions: list[str] = []
        
        if behavior_summary.get("hard_braking_count", 0) >= 2:
            suggestions.append(
                "Maintain greater following distance to reduce hard braking"
            )
        
        if behavior_summary.get("sharp_turn_count", 0) >= 2:
            suggestions.append(
                "Reduce speed before turns for smoother cornering"
            )
        
        if behavior_summary.get("rapid_acceleration_count", 0) >= 2:
            suggestions.append(
                "Accelerate more gradually for better fuel efficiency and comfort"
            )
        
        if behavior_summary.get("over_speed_count", 0) >= 1:
            suggestions.append(
                "Monitor speed limits and maintain appropriate speeds"
            )
        
        if behavior_summary.get("aggressive_driving"):
            suggestions.append(
                "Consider defensive driving techniques for safer journeys"
            )
        
        if not suggestions:
            suggestions.append(
                "Continue maintaining current safe driving practices"
            )
        
        return suggestions
