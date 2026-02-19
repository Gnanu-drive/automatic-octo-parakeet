"""
Prompt Builder Module

This module constructs structured prompts for the LLM to generate
natural language trip narrations.
"""

import json
import logging
from datetime import datetime
from typing import Any

from .models import (
    AnalyzedEvent,
    DriverBehaviorSummary,
    EnrichedCoordinate,
    Phase,
    RouteDescription,
    SemanticSummary,
    SpeedAnomaly,
    TripSummary,
    Turn,
)
from .utils.helpers import format_time


logger = logging.getLogger(__name__)


# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert trip narrator who converts structured vehicle telematics data into engaging, human-readable stories.

Your narration style:
- Write in third person (e.g., "The driver started...", "The vehicle traveled...")
- Use chronological order
- Maintain a natural, storytelling tone
- Include specific times, directions, road names, and locations when available
- Describe speed trends (accelerating, cruising, slowing down)
- Mention significant events (hard braking, sharp turns) in context
- Keep the narration flowing and cohesive
- Be concise but comprehensive

Do NOT:
- Use technical jargon
- Include raw numbers without context
- Make the narration sound like a data report
- Include JSON or structured data in your response
- Add disclaimers or meta-commentary"""


class PromptBuilder:
    """
    Builds structured prompts for LLM trip narration.
    
    Converts semantic summary into a prompt that guides
    the LLM to generate natural language narration.
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        time_format: str = "%I:%M %p",
        date_format: str = "%B %d, %Y",
    ):
        """
        Initialize prompt builder.
        
        Args:
            config: Configuration dictionary
            time_format: strftime format for times
            date_format: strftime format for dates
        """
        self.config = config or {}
        output_config = self.config.get("output", {})
        
        self.time_format = output_config.get("time_format", time_format)
        self.date_format = output_config.get("date_format", date_format)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return SYSTEM_PROMPT
    
    def build_prompt(
        self,
        semantic_summary: SemanticSummary,
        additional_context: str | None = None,
    ) -> str:
        """
        Build the complete prompt for narration generation.
        
        Args:
            semantic_summary: Structured trip summary
            additional_context: Optional markdown with anomaly descriptions
            
        Returns:
            Formatted prompt string
        """
        sections: list[str] = []
        
        # Header
        sections.append("Generate a natural language narration for the following trip:\n")
        
        # Trip overview
        sections.append(self._build_trip_overview(semantic_summary.trip_summary))
        
        # Route description
        sections.append(self._build_route_section(semantic_summary.route_description))
        
        # Trip phases
        sections.append(self._build_phases_section(semantic_summary.phases))
        
        # Turns and direction changes
        if semantic_summary.turns:
            sections.append(self._build_turns_section(semantic_summary.turns))
        
        # Events
        if semantic_summary.events:
            sections.append(self._build_events_section(semantic_summary.events))
        
        # Speed anomalies
        if semantic_summary.anomalies:
            sections.append(self._build_anomalies_section(semantic_summary.anomalies))
        
        # Driver behavior
        sections.append(self._build_behavior_section(semantic_summary.driver_behavior))
        
        # Additional context
        if additional_context:
            sections.append(f"\nAdditional Notes:\n{additional_context}")
        
        # Instructions
        sections.append(self._build_instructions())
        
        return "\n".join(sections)
    
    def _build_trip_overview(self, trip_summary: TripSummary) -> str:
        """Build trip overview section."""
        lines = [
            "=== TRIP OVERVIEW ===",
            f"Trip ID: {trip_summary.trip_id}",
            f"Date: {trip_summary.start_time}",
            f"Duration: {trip_summary.duration}",
            f"Distance: {trip_summary.distance}",
            f"Start: {trip_summary.start_location}",
            f"End: {trip_summary.end_location}",
            f"Average Speed: {trip_summary.average_speed}",
            f"Maximum Speed: {trip_summary.max_speed}",
        ]
        
        if trip_summary.idle_time:
            lines.append(f"Idle Time: {trip_summary.idle_time}")
        
        return "\n".join(lines)
    
    def _build_route_section(self, route_desc: RouteDescription) -> str:
        """Build route description section."""
        lines = [
            "\n=== ROUTE ===",
            f"From: {route_desc.start_location}",
            f"To: {route_desc.end_location}",
            f"Total Distance: {route_desc.total_distance_km:.1f} km",
            f"Route Type: {route_desc.route_type}",
        ]
        
        if route_desc.general_direction:
            lines.append(f"General Direction: {route_desc.general_direction}")
        
        if route_desc.major_roads:
            lines.append(f"Major Roads: {', '.join(route_desc.major_roads[:5])}")
        
        if route_desc.cities_passed:
            lines.append(f"Areas Passed: {', '.join(route_desc.cities_passed[:5])}")
        
        return "\n".join(lines)
    
    def _build_phases_section(self, phases: list[Phase]) -> str:
        """Build trip phases section."""
        if not phases:
            return ""
        
        lines = ["\n=== TRIP PHASES (Chronological) ==="]
        
        for i, phase in enumerate(phases, 1):
            start_time = format_time(phase.start_time, self.time_format)
            
            phase_desc = f"{i}. [{start_time}] {phase.phase_type.value.upper()}"
            
            if phase.start_location:
                phase_desc += f" at {phase.start_location.short_location}"
            
            phase_desc += f" - {phase.description}"
            
            if phase.avg_speed_kmh:
                phase_desc += f" (avg {phase.avg_speed_kmh:.0f} km/h)"
            
            lines.append(phase_desc)
        
        return "\n".join(lines)
    
    def _build_turns_section(self, turns: list[Turn]) -> str:
        """Build turns section."""
        lines = ["\n=== TURNS AND DIRECTION CHANGES ==="]
        
        for turn in turns:
            time_str = format_time(turn.timestamp, self.time_format)
            location = turn.location_name or f"({turn.latitude:.4f}, {turn.longitude:.4f})"
            
            lines.append(
                f"- [{time_str}] {turn.direction.value.upper()} turn "
                f"({turn.angle:.0f}°) {turn.severity} at {location}"
            )
        
        return "\n".join(lines)
    
    def _build_events_section(self, events: list[AnalyzedEvent]) -> str:
        """Build events section."""
        lines = ["\n=== DRIVING EVENTS ==="]
        
        for event in events:
            time_str = format_time(event.timestamp, self.time_format)
            severity = event.severity.value.upper()
            
            lines.append(f"- [{time_str}] [{severity}] {event.description}")
        
        return "\n".join(lines)
    
    def _build_anomalies_section(self, anomalies: list[SpeedAnomaly]) -> str:
        """Build speed anomalies section."""
        lines = ["\n=== SPEED ANOMALIES ==="]
        
        for anomaly in anomalies:
            time_str = format_time(anomaly.timestamp, self.time_format)
            
            lines.append(
                f"- [{time_str}] {anomaly.anomaly_type}: "
                f"{anomaly.speed_kmh:.0f} km/h (expected {anomaly.expected_speed_kmh:.0f} km/h)"
            )
        
        return "\n".join(lines)
    
    def _build_behavior_section(self, behavior: DriverBehaviorSummary) -> str:
        """Build driver behavior section."""
        lines = [
            "\n=== DRIVER BEHAVIOR ===",
            f"Overall Rating: {behavior.overall_rating}/10",
            f"Hard Braking Events: {behavior.hard_braking_count}",
            f"Sharp Turns: {behavior.sharp_turn_count}",
            f"Rapid Accelerations: {behavior.rapid_acceleration_count}",
            f"Speed Limit Compliance: {behavior.speed_compliance:.0f}%",
        ]
        
        if behavior.behavior_summary:
            lines.append(f"Summary: {behavior.behavior_summary}")
        
        if behavior.aggressive_driving:
            lines.append("Note: Aggressive driving patterns detected")
        elif behavior.smooth_driving:
            lines.append("Note: Smooth, controlled driving throughout")
        
        return "\n".join(lines)
    
    def _build_instructions(self) -> str:
        """Build final instructions for the LLM."""
        return """
=== NARRATION INSTRUCTIONS ===
Please generate a natural, flowing narrative that:
1. Starts with the departure (time, location)
2. Describes the journey chronologically
3. Mentions key roads, turns, and landmarks
4. Describes speed patterns naturally (accelerating, cruising, slowing)
5. Includes any notable events in context
6. Ends with the arrival

Write 3-5 paragraphs in a storytelling style. Use third person perspective ("The driver...", "The vehicle...").
"""
    
    def build_simple_prompt(
        self,
        trip_data: dict[str, Any],
    ) -> str:
        """
        Build a simpler prompt directly from trip data dict.
        
        Useful for quick generation without full semantic analysis.
        
        Args:
            trip_data: Dictionary with trip information
            
        Returns:
            Formatted prompt string
        """
        lines = [
            "Generate a natural language narration for this vehicle trip:",
            "",
            f"Start: {trip_data.get('start_location', 'Unknown')} at {trip_data.get('start_time', 'Unknown')}",
            f"End: {trip_data.get('end_location', 'Unknown')} at {trip_data.get('end_time', 'Unknown')}",
            f"Duration: {trip_data.get('duration', 'Unknown')}",
            f"Distance: {trip_data.get('distance', 'Unknown')}",
            "",
        ]
        
        if trip_data.get('roads'):
            lines.append(f"Roads traveled: {', '.join(trip_data['roads'])}")
        
        if trip_data.get('events'):
            lines.append("\nEvents:")
            for event in trip_data['events']:
                lines.append(f"- {event}")
        
        lines.extend([
            "",
            "Write a 2-3 paragraph narration in third person, describing the journey naturally.",
        ])
        
        return "\n".join(lines)
    
    def format_summary_for_context(
        self,
        semantic_summary: SemanticSummary
    ) -> str:
        """
        Format semantic summary as JSON context for the LLM.
        
        Args:
            semantic_summary: The semantic summary
            
        Returns:
            JSON string representation
        """
        # Convert to serializable dict
        summary_dict = {
            "trip": {
                "id": semantic_summary.trip_summary.trip_id,
                "start": semantic_summary.trip_summary.start_location,
                "end": semantic_summary.trip_summary.end_location,
                "duration": semantic_summary.trip_summary.duration,
                "distance": semantic_summary.trip_summary.distance,
            },
            "route": {
                "direction": semantic_summary.route_description.general_direction,
                "type": semantic_summary.route_description.route_type,
                "roads": semantic_summary.route_description.major_roads[:5],
            },
            "phases_count": len(semantic_summary.phases),
            "events_count": len(semantic_summary.events),
            "behavior_score": semantic_summary.driver_behavior.overall_rating,
        }
        
        return json.dumps(summary_dict, indent=2)
