"""
Prompt Builder Module

This module constructs structured prompts for the LLM to generate
natural language trip narrations.

Supports three generation modes:
- narrative: Natural storytelling narration (default)
- navigation_past: Third-person past-tense procedural driving report
- summary: Concise bullet-point trip summary
"""

import json
import logging
from datetime import datetime
from typing import Any

from .models import (
    AnalyzedEvent,
    DriverBehaviorSummary,
    EnrichedCoordinate,
    GenerationMode,
    NavigationAction,
    NavigationInstruction,
    Phase,
    RouteDescription,
    SemanticSummary,
    SpeedAnomaly,
    TripSummary,
    Turn,
    TurnDirection,
)
from .utils.helpers import format_time


logger = logging.getLogger(__name__)


# System prompt for the LLM (narrative mode)
SYSTEM_PROMPT = """You are a mobility analyst and trip narrator.

Write a third-person natural English narration of the trip.

Rules:
- chronological
- human readable
- mention directions
- mention roads/locations
- mention major events
- no bullet points
- paragraph format"""


# System prompt for navigation_past mode
NAVIGATION_PAST_SYSTEM_PROMPT = """You are a driving report generator.
Convert structured navigation instructions into third-person past-tense driving description.
Use short sentences.
Keep Google Maps clarity.
No storytelling.
No assumptions.
No added commentary.
Chronological.
One instruction per line."""


# System prompt for summary mode
SUMMARY_SYSTEM_PROMPT = """You are a trip data summarizer.
Generate a concise bullet-point summary of the trip.
Include only key facts: origin, destination, distance, duration, route highlights, and notable events.
Use bullet points.
Keep it brief and factual.
No narrative or storytelling.
Maximum 10 bullet points."""


# Past-tense action templates
PAST_TENSE_TEMPLATES = {
    NavigationAction.START: "The driver started at {location}.",
    NavigationAction.GO_STRAIGHT: "The driver proceeded {direction} for {distance} along {road}.",
    NavigationAction.TURN_LEFT: "The driver turned left onto {road}.",
    NavigationAction.TURN_RIGHT: "The driver turned right onto {road}.",
    NavigationAction.SLIGHT_LEFT: "The driver took a slight left onto {road}.",
    NavigationAction.SLIGHT_RIGHT: "The driver took a slight right onto {road}.",
    NavigationAction.U_TURN: "The driver made a U-turn onto {road}.",
    NavigationAction.ARRIVE: "The driver arrived at {location}.",
}


class PromptBuilder:
    """
    Builds structured prompts for LLM trip narration.
    
    Converts semantic summary into a prompt that guides
    the LLM to generate natural language narration.
    
    Supports two modes:
    - narrative: Traditional storytelling narration
    - navigation_past: Procedural past-tense driving report
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        time_format: str = "%I:%M %p",
        date_format: str = "%B %d, %Y",
        mode: GenerationMode | str = GenerationMode.NARRATIVE,
    ):
        """
        Initialize prompt builder.
        
        Args:
            config: Configuration dictionary
            time_format: strftime format for times
            date_format: strftime format for dates
            mode: Generation mode (narrative or navigation_past)
        """
        self.config = config or {}
        output_config = self.config.get("output", {})
        
        self.time_format = output_config.get("time_format", time_format)
        self.date_format = output_config.get("date_format", date_format)
        
        # Set generation mode
        if isinstance(mode, str):
            self.mode = GenerationMode(mode)
        else:
            self.mode = mode
    
    def set_mode(self, mode: GenerationMode | str) -> None:
        """Set the generation mode."""
        if isinstance(mode, str):
            self.mode = GenerationMode(mode)
        else:
            self.mode = mode

    def get_system_prompt(self) -> str:
        """Return system prompt for LLM based on current mode."""
        if self.mode == GenerationMode.NAVIGATION_PAST:
            return NAVIGATION_PAST_SYSTEM_PROMPT
        
        if self.mode == GenerationMode.SUMMARY:
            return SUMMARY_SYSTEM_PROMPT
        
        # Default narrative mode prompt
        return """You are a trip data narrator. Your task is to describe vehicle trips in a factual, objective manner.

        Rules:
        - Use third-person perspective only (e.g., "The driver", "The vehicle")
        - State facts without emotional language or subjective interpretations
        - Do not use phrases like "smoothly", "carefully", "impressively", "unfortunately"
        - Avoid dramatic descriptions or storytelling elements
        - Report events, distances, times, and speeds as plain data
        - Keep sentences short and informative
        - Do not add commentary or opinions about driving quality
        - Do not use exclamation marks or rhetorical questions

        Example of what NOT to write:
        "The driver skillfully navigated through challenging traffic conditions."

        Example of correct style:
        "The driver traveled through traffic on Main Street for 2.3 km."
    """
    # def get_system_prompt(self) -> str:
    #     """Get the system prompt for the LLM."""
    #     return SYSTEM_PROMPT
    
    def build_prompt(
        self,
        semantic_summary: SemanticSummary,
        additional_context: str | None = None,
        navigation_segments: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build the complete prompt for narration generation.
        
        Args:
            semantic_summary: Structured trip summary
            additional_context: Optional markdown with anomaly descriptions
            navigation_segments: Navigation segments for navigation_past mode
            
        Returns:
            Formatted prompt string with JSON input
        """
        if self.mode == GenerationMode.NAVIGATION_PAST:
            return self._build_navigation_past_prompt(
                semantic_summary, navigation_segments
            )
        
        if self.mode == GenerationMode.SUMMARY:
            return self._build_summary_prompt(semantic_summary)
        
        # Default narrative mode
        # Build structured JSON summary
        trip_json = self._build_trip_json(semantic_summary)
        
        # Add additional context if provided
        if additional_context:
            trip_json["additional_notes"] = additional_context
        
        # Format as clean JSON
        json_str = json.dumps(trip_json, indent=2, default=str)
        
        return f"Input:\n{json_str}"
    
    def _build_summary_prompt(
        self,
        semantic_summary: SemanticSummary,
    ) -> str:
        """
        Build prompt for summary mode.
        
        Creates a concise data structure for bullet-point summary generation.
        """
        ts = semantic_summary.trip_summary
        rd = semantic_summary.route_description
        db = semantic_summary.driver_behavior
        
        summary_data = {
            "trip_id": ts.trip_id,
            "origin": ts.start_location,
            "destination": ts.end_location,
            "distance": ts.distance,
            "duration": ts.duration,
            "start_time": ts.start_time,
            "end_time": ts.end_time,
            "average_speed": ts.average_speed,
            "max_speed": ts.max_speed,
            "route_type": rd.route_type,
            "direction": rd.general_direction,
            "major_roads": rd.major_roads[:3] if rd.major_roads else [],
            "event_count": ts.event_count,
            "driver_rating": f"{db.overall_rating}/10",
        }
        
        prompt_parts = [
            "Generate a bullet-point summary of this trip:",
            "",
            json.dumps(summary_data, indent=2, default=str),
            "",
            "Format: Use bullet points (•). Maximum 10 points. Include only key facts.",
        ]
        
        return "\n".join(prompt_parts)
    
    def render_summary_report(
        self,
        semantic_summary: SemanticSummary,
    ) -> str:
        """
        Render a summary report without LLM (fallback).
        
        Args:
            semantic_summary: Structured trip summary
            
        Returns:
            Bullet-point summary string
        """
        ts = semantic_summary.trip_summary
        rd = semantic_summary.route_description
        db = semantic_summary.driver_behavior
        
        lines = [
            f"• Origin: {ts.start_location}",
            f"• Destination: {ts.end_location}",
            f"• Distance: {ts.distance}",
            f"• Duration: {ts.duration}",
            f"• Start Time: {ts.start_time}",
            f"• Average Speed: {ts.average_speed}",
            f"• Max Speed: {ts.max_speed}",
            f"• Route Type: {rd.route_type.title()}",
            f"• Direction: {rd.general_direction or 'N/A'}",
        ]
        
        if rd.major_roads:
            lines.append(f"• Major Roads: {', '.join(rd.major_roads[:3])}")
        
        if ts.event_count > 0:
            lines.append(f"• Events: {ts.event_count} recorded")
        
        lines.append(f"• Driver Rating: {db.overall_rating}/10")
        
        return "\n".join(lines)
    
    def _build_navigation_past_prompt(
        self,
        semantic_summary: SemanticSummary,
        navigation_segments: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build prompt for navigation_past mode.
        
        Constructs structured navigation instructions and formats
        them for the LLM to convert into past-tense driving report.
        """
        instructions = self.build_navigation_instructions(
            semantic_summary, navigation_segments
        )
        
        # Convert instructions to JSON format
        instructions_json = [inst.to_dict() for inst in instructions]
        
        prompt_parts = [
            "Convert the following navigation instructions into a third-person past-tense driving report.",
            "Each instruction should be one line. Use the exact format shown in examples.",
            "",
            "Instructions:",
            json.dumps(instructions_json, indent=2),
        ]
        
        return "\n".join(prompt_parts)
    
    def build_navigation_instructions(
        self,
        semantic_summary: SemanticSummary,
        navigation_segments: list[dict[str, Any]] | None = None,
    ) -> list[NavigationInstruction]:
        """
        Build structured navigation instructions from semantic summary.
        
        Args:
            semantic_summary: Structured trip summary
            navigation_segments: Pre-computed navigation segments
            
        Returns:
            List of NavigationInstruction objects
        """
        instructions: list[NavigationInstruction] = []
        
        # Start instruction
        start_location = semantic_summary.trip_summary.start_location
        instructions.append(NavigationInstruction(
            action=NavigationAction.START,
            location=start_location,
        ))
        
        # Process navigation segments if provided
        if navigation_segments:
            for segment in navigation_segments:
                movement = segment.get("movement_type", "straight")
                distance = segment.get("distance_m", 0)
                road = segment.get("road_name")
                direction = segment.get("direction")
                
                action = self._movement_to_action(movement)
                
                if action == NavigationAction.GO_STRAIGHT:
                    if distance > 0:
                        instructions.append(NavigationInstruction(
                            action=action,
                            distance_m=distance,
                            road=road or "the road",
                            direction=direction or "forward",
                        ))
                else:
                    # Turn instruction
                    instructions.append(NavigationInstruction(
                        action=action,
                        road=road or "the road",
                    ))
        else:
            # Fallback: use turns from semantic summary
            self._build_instructions_from_turns(
                semantic_summary, instructions
            )
        
        # Arrive instruction
        end_location = semantic_summary.trip_summary.end_location
        instructions.append(NavigationInstruction(
            action=NavigationAction.ARRIVE,
            location=end_location,
        ))
        
        return instructions
    
    def _movement_to_action(self, movement_type: str) -> NavigationAction:
        """Convert movement type string to NavigationAction."""
        mapping = {
            "straight": NavigationAction.GO_STRAIGHT,
            "turn_left": NavigationAction.TURN_LEFT,
            "turn_right": NavigationAction.TURN_RIGHT,
            "slight_left": NavigationAction.SLIGHT_LEFT,
            "slight_right": NavigationAction.SLIGHT_RIGHT,
            "u_turn": NavigationAction.U_TURN,
        }
        return mapping.get(movement_type, NavigationAction.GO_STRAIGHT)
    
    def _build_instructions_from_turns(
        self,
        semantic_summary: SemanticSummary,
        instructions: list[NavigationInstruction],
    ) -> None:
        """Build navigation instructions from turns when segments not available."""
        enriched_route = semantic_summary.enriched_route
        turns = semantic_summary.turns
        
        if not enriched_route:
            return
        
        # Get general direction
        general_direction = semantic_summary.route_description.general_direction or "forward"
        
        # Calculate total distance
        total_distance = semantic_summary.route_description.total_distance_km * 1000
        
        if not turns:
            # No turns: single straight segment
            major_roads = semantic_summary.route_description.major_roads
            road = major_roads[0] if major_roads else "the road"
            
            instructions.append(NavigationInstruction(
                action=NavigationAction.GO_STRAIGHT,
                distance_m=total_distance,
                road=road,
                direction=general_direction,
            ))
            return
        
        # With turns: create segments between turns
        prev_distance = 0
        turn_count = len(turns)
        distance_per_segment = total_distance / (turn_count + 1) if turn_count > 0 else total_distance
        
        for i, turn in enumerate(turns):
            # Add straight segment before turn
            road = turn.location_name or "the road"
            
            instructions.append(NavigationInstruction(
                action=NavigationAction.GO_STRAIGHT,
                distance_m=distance_per_segment,
                road=road,
                direction=general_direction,
            ))
            
            # Add turn instruction
            turn_action = self._turn_direction_to_action(turn.direction, turn.angle)
            instructions.append(NavigationInstruction(
                action=turn_action,
                road=road,
            ))
    
    def _turn_direction_to_action(
        self,
        direction: TurnDirection,
        angle: float,
    ) -> NavigationAction:
        """Convert turn direction to navigation action."""
        if direction == TurnDirection.U_TURN:
            return NavigationAction.U_TURN
        elif direction == TurnDirection.LEFT:
            if angle < 45:
                return NavigationAction.SLIGHT_LEFT
            return NavigationAction.TURN_LEFT
        elif direction == TurnDirection.RIGHT:
            if angle < 45:
                return NavigationAction.SLIGHT_RIGHT
            return NavigationAction.TURN_RIGHT
        return NavigationAction.GO_STRAIGHT
    
    def format_distance_human(self, distance_m: float) -> str:
        """
        Format distance in human-friendly format.
        
        <1000m → meters
        ≥1000m → kilometers (1 decimal precision)
        
        Args:
            distance_m: Distance in meters
            
        Returns:
            Human-friendly distance string
        """
        if distance_m < 1000:
            return f"{int(round(distance_m))} meters"
        return f"{distance_m / 1000:.1f} kilometers"
    
    def render_navigation_report(
        self,
        instructions: list[NavigationInstruction],
    ) -> str:
        """
        Render navigation instructions as past-tense driving report.
        
        This is a fallback renderer that doesn't require LLM.
        
        Args:
            instructions: List of navigation instructions
            
        Returns:
            Past-tense driving report string
        """
        lines: list[str] = []
        
        for inst in instructions:
            line = self._render_instruction(inst)
            if line:
                lines.append(line)
        
        return "\n".join(lines)
    
    def _render_instruction(self, inst: NavigationInstruction) -> str:
        """Render a single navigation instruction to past-tense text."""
        template = PAST_TENSE_TEMPLATES.get(inst.action)
        if not template:
            return ""
        
        # Build substitution dict
        subs: dict[str, str] = {}
        
        if inst.location:
            subs["location"] = inst.location
        if inst.road:
            subs["road"] = inst.road
        if inst.direction:
            subs["direction"] = inst.direction
        if inst.distance_m is not None:
            subs["distance"] = self.format_distance_human(inst.distance_m)
        
        # Handle missing values with defaults
        if "{location}" in template and "location" not in subs:
            subs["location"] = "the location"
        if "{road}" in template and "road" not in subs:
            subs["road"] = "the road"
        if "{direction}" in template and "direction" not in subs:
            subs["direction"] = "forward"
        if "{distance}" in template and "distance" not in subs:
            subs["distance"] = "some distance"
        
        try:
            return template.format(**subs)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            return ""
    
    def _build_trip_json(self, semantic_summary: SemanticSummary) -> dict[str, Any]:
        """Build structured JSON from semantic summary."""
        ts = semantic_summary.trip_summary
        rd = semantic_summary.route_description
        db = semantic_summary.driver_behavior
        
        trip_json: dict[str, Any] = {
            "trip": {
                "id": ts.trip_id,
                "start_time": ts.start_time,
                "end_time": ts.end_time,
                "duration": ts.duration,
                "distance": ts.distance,
            },
            "route": {
                "start": ts.start_location,
                "end": ts.end_location,
                "direction": rd.general_direction,
                "type": rd.route_type,
                "major_roads": rd.major_roads[:5] if rd.major_roads else [],
                "areas": rd.cities_passed[:5] if rd.cities_passed else [],
            },
            "speed": {
                "average": ts.average_speed,
                "maximum": ts.max_speed,
            },
            "driver_behavior": {
                "rating": f"{db.overall_rating}/10",
                "hard_braking": db.hard_braking_count,
                "sharp_turns": db.sharp_turn_count,
                "rapid_acceleration": db.rapid_acceleration_count,
                "style": "smooth" if db.smooth_driving else "aggressive" if db.aggressive_driving else "normal",
            },
        }
        
        # Add phases (chronological journey)
        if semantic_summary.phases:
            trip_json["phases"] = [
                {
                    "time": format_time(p.start_time, self.time_format),
                    "type": p.phase_type.value,
                    "location": p.start_location.short_location if p.start_location else None,
                    "description": p.description,
                    "avg_speed_kmh": round(p.avg_speed_kmh) if p.avg_speed_kmh else None,
                }
                for p in semantic_summary.phases
            ]
        
        # Add events
        if semantic_summary.events:
            trip_json["events"] = [
                {
                    "time": format_time(e.timestamp, self.time_format),
                    "type": e.event_type,
                    "severity": e.severity.value,
                    "description": e.description,
                }
                for e in semantic_summary.events
            ]
        
        # Add turns
        if semantic_summary.turns:
            trip_json["turns"] = [
                {
                    "time": format_time(t.timestamp, self.time_format),
                    "direction": t.direction.value,
                    "angle": round(t.angle),
                    "location": t.location_name,
                }
                for t in semantic_summary.turns[:10]  # Limit to 10 turns
            ]
        
        return trip_json
    
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
