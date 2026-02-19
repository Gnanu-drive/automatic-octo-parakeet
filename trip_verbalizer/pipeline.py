"""
Pipeline Module

This module orchestrates the complete trip verbalization pipeline,
coordinating all analysis stages and LLM integration.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .events import EventAnalyzer
from .geo import GeoEnricher
from .llm_client import LLMClient, MockLLMClient, LLMError, FallbackNarrator, LLMConnectionError
from .models import (
    AnalyzedEvent,
    Coordinate,
    DriverBehaviorSummary,
    EnrichedCoordinate,
    NarrationOutput,
    Phase,
    RouteDescription,
    SemanticSummary,
    SpeedAnomaly,
    TripData,
    TripSummary,
    Turn,
)
from .prompt_builder import PromptBuilder
from .spatial import SpatialAnalyzer
from .temporal import TemporalAnalyzer
from .utils.helpers import (
    format_distance,
    format_duration,
    format_speed,
    format_time,
    load_config,
)


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Exception raised when pipeline execution fails."""
    pass


class TripVerbalizerPipeline:
    """
    Main pipeline orchestrator for trip verbalization.
    
    Coordinates all stages:
    1. Data Loading & Validation
    2. Geo Enrichment
    3. Spatial Analysis
    4. Temporal Analysis
    5. Event Analysis
    6. Semantic Structuring
    7. Prompt Building
    8. LLM Generation
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        use_mock_llm: bool = False,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary (overrides config file)
            config_path: Path to config.yaml file
            use_mock_llm: If True, use mock LLM for testing
        """
        # Load configuration
        if config:
            self.config = config
        else:
            try:
                self.config = load_config(config_path)
            except FileNotFoundError:
                logger.warning("Config file not found, using defaults")
                self.config = {}
        
        # Initialize analyzers
        self.geo_enricher = GeoEnricher(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.event_analyzer = EventAnalyzer(self.config)
        self.prompt_builder = PromptBuilder(self.config)
        
        # Initialize LLM client
        self.use_mock_llm = use_mock_llm
        if use_mock_llm:
            self.llm_client = MockLLMClient.from_config(self.config)
        else:
            self.llm_client = LLMClient.from_config(self.config)
        
        # Debug logs storage
        self._debug_logs: list[str] = []
        self._warnings: list[str] = []
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log message and store for debug output."""
        self._debug_logs.append(f"[{level.upper()}] {message}")
        getattr(logger, level)(message)
    
    def _warn(self, message: str) -> None:
        """Log warning and store."""
        self._warnings.append(message)
        logger.warning(message)
    
    async def process(
        self,
        trip_data: TripData,
        additional_context: str | None = None,
        geocode_sample_rate: int = 5,
    ) -> NarrationOutput:
        """
        Process trip data through the complete pipeline.
        
        Args:
            trip_data: Validated trip data input
            additional_context: Optional markdown with anomaly notes
            geocode_sample_rate: Geocode every Nth point (1=all)
            
        Returns:
            NarrationOutput with narration and metadata
        """
        start_time = time.time()
        self._debug_logs = []
        self._warnings = []
        
        self._log(f"Starting pipeline for trip {trip_data.trip_id}")
        
        try:
            # Stage 1: Validate input
            self._log("Stage 1: Validating input data")
            self._validate_input(trip_data)
            
            # Stage 2: Geo Enrichment
            self._log("Stage 2: Geo enrichment")
            enriched_route = await self._enrich_coordinates(
                trip_data.route,
                geocode_sample_rate
            )
            
            # Stage 3: Spatial Analysis
            self._log("Stage 3: Spatial analysis")
            enriched_route = self.spatial_analyzer.compute_bearings(enriched_route)
            turns = self.spatial_analyzer.detect_turns(enriched_route)
            
            # Stage 4: Temporal Analysis
            self._log("Stage 4: Temporal analysis")
            phases = self.temporal_analyzer.segment_trip_phases(
                enriched_route,
                trip_data.speed_data
            )
            anomalies = self.temporal_analyzer.detect_speed_anomalies(
                enriched_route,
                trip_data.speed_data
            )
            
            # Stage 5: Event Analysis
            self._log("Stage 5: Event analysis")
            analyzed_events = self.event_analyzer.analyze_events(
                trip_data.events,
                enriched_route,
                trip_data.speed_data
            )
            
            # Stage 6: Semantic Structuring
            self._log("Stage 6: Building semantic summary")
            semantic_summary = self._build_semantic_summary(
                trip_data,
                enriched_route,
                phases,
                turns,
                analyzed_events,
                anomalies,
            )
            
            # Stage 7: Prompt Building
            self._log("Stage 7: Constructing LLM prompt")
            prompt = self.prompt_builder.build_prompt(
                semantic_summary,
                additional_context
            )
            
            # Stage 8: LLM Generation
            self._log("Stage 8: Generating narration with LLM")
            narration = await self._generate_narration(prompt)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self._log(f"Pipeline completed in {processing_time:.2f}s")
            
            return NarrationOutput(
                narration=narration,
                semantic_summary=semantic_summary,
                generated_at=datetime.utcnow(),
                model_used=self.llm_client.model if not self.use_mock_llm else "mock",
                processing_time_seconds=processing_time,
                debug_logs=self._debug_logs,
                warnings=self._warnings,
            )
            
        except Exception as e:
            self._log(f"Pipeline failed: {e}", level="error")
            raise PipelineError(f"Pipeline execution failed: {e}") from e
    
    def _validate_input(self, trip_data: TripData) -> None:
        """Validate trip data input."""
        if len(trip_data.route) < 2:
            raise PipelineError("Trip must have at least 2 route points")
        
        # Check for valid timestamps
        for i, coord in enumerate(trip_data.route):
            if coord.timestamp is None:
                raise PipelineError(f"Route point {i} missing timestamp")
        
        # Check timestamps are in order
        for i in range(1, len(trip_data.route)):
            if trip_data.route[i].timestamp < trip_data.route[i-1].timestamp:
                self._warn(f"Timestamps not in order at point {i}")
    
    async def _enrich_coordinates(
        self,
        coordinates: list[Coordinate],
        sample_rate: int = 5
    ) -> list[EnrichedCoordinate]:
        """Enrich coordinates with geocoding data."""
        async with self.geo_enricher:
            return await self.geo_enricher.enrich_coordinates(
                coordinates,
                sample_rate=sample_rate
            )
    
    def _build_semantic_summary(
        self,
        trip_data: TripData,
        enriched_route: list[EnrichedCoordinate],
        phases: list[Phase],
        turns: list[Turn],
        events: list[AnalyzedEvent],
        anomalies: list[SpeedAnomaly],
    ) -> SemanticSummary:
        """Build the complete semantic summary."""
        # Build trip summary
        trip_stats = self.temporal_analyzer.compute_trip_statistics(
            enriched_route,
            trip_data.speed_data
        )
        
        trip_summary = TripSummary(
            trip_id=trip_data.trip_id,
            duration=format_duration(trip_stats.get("total_duration_seconds", 0)),
            distance=format_distance(
                self.spatial_analyzer.compute_total_distance(enriched_route)
            ),
            start_time=format_time(trip_data.start_time) if trip_data.start_time else "Unknown",
            end_time=format_time(trip_data.end_time) if trip_data.end_time else "Unknown",
            start_location=self._get_location_string(enriched_route[0] if enriched_route else None),
            end_location=self._get_location_string(enriched_route[-1] if enriched_route else None),
            average_speed=format_speed(trip_stats.get("average_moving_speed_kmh", 0)),
            max_speed=format_speed(trip_stats.get("max_speed_kmh", 0)),
            event_count=len(events),
            idle_time=format_duration(trip_stats.get("idle_time_seconds", 0)) if trip_stats.get("idle_time_seconds", 0) > 0 else None,
        )
        
        # Build route description
        route_description = RouteDescription(
            start_location=trip_summary.start_location,
            end_location=trip_summary.end_location,
            total_distance_km=self.spatial_analyzer.compute_total_distance(enriched_route) / 1000,
            major_roads=self.spatial_analyzer.get_major_roads(enriched_route),
            cities_passed=self.spatial_analyzer.get_cities_passed(enriched_route),
            general_direction=self.spatial_analyzer.compute_general_direction(enriched_route),
            route_type=self._determine_route_type(enriched_route),
        )
        
        # Build driver behavior summary
        behavior_stats = self.event_analyzer.compute_behavior_summary(events)
        driver_behavior = DriverBehaviorSummary(
            overall_rating=behavior_stats.get("overall_score", 10.0),
            hard_braking_count=behavior_stats.get("hard_braking_count", 0),
            sharp_turn_count=behavior_stats.get("sharp_turn_count", 0),
            rapid_acceleration_count=behavior_stats.get("rapid_acceleration_count", 0),
            over_speed_count=behavior_stats.get("over_speed_count", 0),
            aggressive_driving=behavior_stats.get("aggressive_driving", False),
            smooth_driving=behavior_stats.get("smooth_driving", True),
            behavior_summary=self.event_analyzer.generate_behavior_narrative(behavior_stats),
            improvement_suggestions=self.event_analyzer.get_improvement_suggestions(behavior_stats),
        )
        
        # Build notable observations
        observations = self._generate_observations(
            enriched_route, phases, events, anomalies
        )
        
        return SemanticSummary(
            trip_summary=trip_summary,
            phases=phases,
            events=events,
            route_description=route_description,
            driver_behavior=driver_behavior,
            notable_observations=observations,
            anomalies=anomalies,
            enriched_route=enriched_route,
            turns=turns,
        )
    
    def _get_location_string(
        self,
        coord: EnrichedCoordinate | None
    ) -> str:
        """Get human-readable location string."""
        if not coord:
            return "Unknown"
        
        if coord.location:
            return coord.location.short_location
        
        return f"{coord.latitude:.4f}, {coord.longitude:.4f}"
    
    def _determine_route_type(
        self,
        enriched_route: list[EnrichedCoordinate]
    ) -> str:
        """Determine if route is urban, highway, rural, or mixed."""
        # Simple heuristic based on speed patterns
        if not enriched_route:
            return "unknown"
        
        speeds = [c.speed_kmh for c in enriched_route if c.speed_kmh]
        if not speeds:
            return "mixed"
        
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        
        if max_speed > 100 and avg_speed > 60:
            return "highway"
        elif avg_speed < 40:
            return "urban"
        else:
            return "mixed"
    
    def _generate_observations(
        self,
        enriched_route: list[EnrichedCoordinate],
        phases: list[Phase],
        events: list[AnalyzedEvent],
        anomalies: list[SpeedAnomaly],
    ) -> list[str]:
        """Generate notable observations from analysis."""
        observations: list[str] = []
        
        # Route observations
        geometry = self.spatial_analyzer.analyze_route_geometry(enriched_route)
        if geometry.get("is_circular"):
            observations.append("Route was circular, returning to the starting point")
        if geometry.get("straightness_ratio", 0) > 0.95:
            observations.append("Route was very direct with minimal deviations")
        
        # Phase observations
        idle_phases = [p for p in phases if p.phase_type.value == "idle"]
        if len(idle_phases) > 2:
            observations.append(f"Multiple stops/idle periods detected ({len(idle_phases)})")
        
        # Event observations
        severe_events = [e for e in events if e.severity.value in ["severe", "critical"]]
        if severe_events:
            observations.append(f"{len(severe_events)} severe driving events recorded")
        
        # Anomaly observations
        if anomalies:
            spike_count = sum(1 for a in anomalies if a.anomaly_type == "speed_spike")
            drop_count = sum(1 for a in anomalies if a.anomaly_type == "sudden_drop")
            if spike_count:
                observations.append(f"{spike_count} sudden speed increase(s) detected")
            if drop_count:
                observations.append(f"{drop_count} sudden braking/slowdown(s) detected")
        
        return observations
    
    async def _generate_narration(self, prompt: str) -> str:
        """Generate narration using LLM."""
        # Check LLM availability
        if not self.use_mock_llm:
            is_available = await self.llm_client.health_check()
            if not is_available:
                self._warn("LLM server not available, using fallback narration")
                return self._generate_fallback_narration()
        
        try:
            async with self.llm_client:
                system_prompt = self.prompt_builder.get_system_prompt()
                narration = await self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                return narration
        except LLMError as e:
            self._warn(f"LLM generation failed: {e}")
            return self._generate_fallback_narration()
    
    def _generate_fallback_narration(self) -> str:
        """Generate fallback narration when LLM is unavailable."""
        fallback = FallbackNarrator()
        
        # Try to extract useful info for better fallback
        try:
            distance_km = getattr(self, '_last_distance_km', 0)
            duration_min = getattr(self, '_last_duration_min', 0)
            start_loc = getattr(self, '_last_start_location', 'unknown')
            end_loc = getattr(self, '_last_end_location', 'unknown')
            avg_speed = getattr(self, '_last_avg_speed', 0)
            events = getattr(self, '_last_events', [])
            
            return fallback.narrate(
                distance_km=distance_km,
                duration_min=duration_min,
                start_location=start_loc,
                end_location=end_loc,
                avg_speed_kmh=avg_speed,
                events=events,
            )
        except Exception:
            return (
                "The driver completed a journey from the starting location to the destination. "
                "The trip proceeded through various phases including acceleration, cruising, and stops. "
                "For detailed analysis, please review the structured metadata."
            )
    
    @classmethod
    async def from_json_file(
        cls,
        json_path: str | Path,
        config_path: str | Path | None = None,
        use_mock_llm: bool = False,
    ) -> NarrationOutput:
        """
        Convenience method to process a trip JSON file.
        
        Args:
            json_path: Path to trip JSON file
            config_path: Optional config file path
            use_mock_llm: If True, use mock LLM
            
        Returns:
            NarrationOutput
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Trip JSON not found: {json_path}")
        
        with open(json_path, "r") as f:
            trip_dict = json.load(f)
        
        trip_data = TripData.model_validate(trip_dict)
        
        pipeline = cls(
            config_path=config_path,
            use_mock_llm=use_mock_llm
        )
        
        return await pipeline.process(trip_data)
    
    @classmethod
    def process_sync(
        cls,
        trip_data: TripData,
        config: dict[str, Any] | None = None,
        use_mock_llm: bool = False,
    ) -> NarrationOutput:
        """
        Synchronous wrapper for pipeline processing.
        
        Args:
            trip_data: Validated trip data
            config: Configuration dictionary
            use_mock_llm: If True, use mock LLM
            
        Returns:
            NarrationOutput
        """
        pipeline = cls(config=config, use_mock_llm=use_mock_llm)
        return asyncio.run(pipeline.process(trip_data))
