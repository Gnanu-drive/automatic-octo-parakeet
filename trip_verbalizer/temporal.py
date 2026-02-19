"""
Temporal Analysis Module

This module handles time-based analysis of trip data including:
- Trip phase segmentation (start, acceleration, cruising, deceleration, stop)
- Speed pattern analysis
- Idle time detection
- Speed anomaly detection
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .models import (
    Coordinate,
    EnrichedCoordinate,
    Phase,
    PhaseType,
    SpeedAnomaly,
    SpeedData,
    EventSeverity,
)
from .utils.helpers import format_duration, format_speed


logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Performs temporal analysis on trip data.
    
    Segments trips into phases and detects speed anomalies.
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize temporal analyzer.
        
        Args:
            config: Configuration dictionary with temporal settings
        """
        self.config = config or {}
        temporal_config = self.config.get("temporal", {})
        
        # Speed thresholds (km/h)
        self.idle_threshold = temporal_config.get("idle_threshold", 2)
        self.low_speed_threshold = temporal_config.get("low_speed_threshold", 20)
        self.cruising_min_speed = temporal_config.get("cruising_min_speed", 40)
        self.high_speed_threshold = temporal_config.get("high_speed_threshold", 100)
        
        # Acceleration thresholds (km/h per second)
        self.acceleration_mild = temporal_config.get("acceleration_mild", 2.0)
        self.acceleration_moderate = temporal_config.get("acceleration_moderate", 5.0)
        self.acceleration_hard = temporal_config.get("acceleration_hard", 8.0)
        
        # Phase settings
        self.min_phase_duration = temporal_config.get("min_phase_duration", 5)
        
        # Anomaly detection
        self.speed_spike_factor = temporal_config.get("speed_spike_factor", 1.5)
        self.idle_anomaly_threshold = temporal_config.get("idle_anomaly_threshold", 120)
    
    def segment_trip_phases(
        self,
        coordinates: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> list[Phase]:
        """
        Segment the trip into distinct phases.
        
        Args:
            coordinates: List of enriched coordinates
            speed_data: Optional speed data (computed if not provided)
            
        Returns:
            List of trip phases
        """
        if len(coordinates) < 2:
            return []
        
        # Compute speeds if not provided
        speeds = self._compute_speeds(coordinates, speed_data)
        
        if not speeds:
            return []
        
        # Classify each point
        point_phases = self._classify_points(coordinates, speeds)
        
        # Merge consecutive points into phases
        phases = self._merge_phases(coordinates, point_phases)
        
        # Filter out very short phases
        phases = self._filter_short_phases(phases)
        
        logger.info(f"Segmented trip into {len(phases)} phases")
        return phases
    
    def _compute_speeds(
        self,
        coordinates: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> list[float]:
        """Compute speeds at each coordinate."""
        if speed_data and len(speed_data) >= len(coordinates):
            # Use provided speed data
            return [s.speed_kmh for s in speed_data[:len(coordinates)]]
        
        # Compute speeds from coordinates
        speeds: list[float] = []
        
        for i in range(len(coordinates)):
            if i == 0:
                speeds.append(0.0)
                continue
            
            prev = coordinates[i - 1]
            curr = coordinates[i]
            
            # Calculate time delta in hours
            time_delta = (curr.timestamp - prev.timestamp).total_seconds() / 3600
            
            if time_delta <= 0:
                speeds.append(speeds[-1] if speeds else 0.0)
                continue
            
            # Calculate distance in km
            if curr.distance_from_prev:
                distance_km = curr.distance_from_prev / 1000
            else:
                # Fallback calculation
                from .utils.helpers import haversine_distance
                distance_km = haversine_distance(
                    prev.latitude, prev.longitude,
                    curr.latitude, curr.longitude
                ) / 1000
            
            # Speed in km/h
            speed = distance_km / time_delta
            speeds.append(speed)
        
        return speeds
    
    def _classify_points(
        self,
        coordinates: list[EnrichedCoordinate],
        speeds: list[float]
    ) -> list[PhaseType]:
        """Classify each point into a phase type."""
        classifications: list[PhaseType] = []
        
        for i, speed in enumerate(speeds):
            if speed <= self.idle_threshold:
                phase_type = PhaseType.IDLE if i > 0 else PhaseType.START
            elif speed <= self.low_speed_threshold:
                # Determine if accelerating or decelerating
                if i > 0 and speeds[i - 1] < speed:
                    phase_type = PhaseType.ACCELERATION
                elif i > 0 and speeds[i - 1] > speed:
                    phase_type = PhaseType.DECELERATION
                else:
                    phase_type = PhaseType.CRUISING
            elif speed >= self.cruising_min_speed:
                phase_type = PhaseType.CRUISING
            else:
                # Transition speed - check acceleration
                if i > 0:
                    accel = self._compute_acceleration(speeds, i)
                    if accel > self.acceleration_mild:
                        phase_type = PhaseType.ACCELERATION
                    elif accel < -self.acceleration_mild:
                        phase_type = PhaseType.DECELERATION
                    else:
                        phase_type = PhaseType.CRUISING
                else:
                    phase_type = PhaseType.START
            
            classifications.append(phase_type)
        
        # Mark first point as START and last as STOP
        if classifications:
            classifications[0] = PhaseType.START
            if speeds[-1] <= self.idle_threshold:
                classifications[-1] = PhaseType.STOP
        
        return classifications
    
    def _compute_acceleration(
        self,
        speeds: list[float],
        index: int,
        window: int = 3
    ) -> float:
        """Compute acceleration at a point (km/h per second)."""
        if index < 1:
            return 0.0
        
        # Use average over window
        start_idx = max(0, index - window)
        
        speed_change = speeds[index] - speeds[start_idx]
        # Assume 1 second between points for simplicity
        # In production, use actual timestamps
        time_seconds = (index - start_idx)
        
        return speed_change / time_seconds if time_seconds > 0 else 0.0
    
    def _merge_phases(
        self,
        coordinates: list[EnrichedCoordinate],
        point_phases: list[PhaseType]
    ) -> list[Phase]:
        """Merge consecutive points with same phase type."""
        phases: list[Phase] = []
        
        if not point_phases:
            return phases
        
        current_type = point_phases[0]
        start_idx = 0
        
        for i in range(1, len(point_phases)):
            if point_phases[i] != current_type:
                # Create phase from start_idx to i-1
                phase = self._create_phase(
                    coordinates,
                    point_phases,
                    start_idx,
                    i - 1,
                    current_type
                )
                phases.append(phase)
                
                current_type = point_phases[i]
                start_idx = i
        
        # Add final phase
        phase = self._create_phase(
            coordinates,
            point_phases,
            start_idx,
            len(point_phases) - 1,
            current_type
        )
        phases.append(phase)
        
        return phases
    
    def _create_phase(
        self,
        coordinates: list[EnrichedCoordinate],
        point_phases: list[PhaseType],
        start_idx: int,
        end_idx: int,
        phase_type: PhaseType
    ) -> Phase:
        """Create a Phase object from indices."""
        start_coord = coordinates[start_idx]
        end_coord = coordinates[end_idx]
        
        # Calculate duration
        duration = (end_coord.timestamp - start_coord.timestamp).total_seconds()
        
        # Calculate distance
        distance = 0.0
        for i in range(start_idx + 1, end_idx + 1):
            if coordinates[i].distance_from_prev:
                distance += coordinates[i].distance_from_prev
        
        # Calculate speeds
        speeds = self._compute_speeds(coordinates[start_idx:end_idx + 1], None)
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        min_speed = min(speeds) if speeds else 0.0
        
        # Generate description
        description = self._generate_phase_description(phase_type, duration, avg_speed)
        
        return Phase(
            phase_type=phase_type,
            start_time=start_coord.timestamp,
            end_time=end_coord.timestamp,
            start_location=start_coord.location,
            end_location=end_coord.location,
            duration_seconds=duration,
            distance_meters=distance,
            avg_speed_kmh=avg_speed,
            max_speed_kmh=max_speed,
            min_speed_kmh=min_speed,
            description=description,
        )
    
    def _generate_phase_description(
        self,
        phase_type: PhaseType,
        duration: float,
        avg_speed: float
    ) -> str:
        """Generate human-readable phase description."""
        duration_str = format_duration(duration)
        speed_str = format_speed(avg_speed)
        
        descriptions = {
            PhaseType.START: f"Started the journey",
            PhaseType.ACCELERATION: f"Accelerated for {duration_str}",
            PhaseType.CRUISING: f"Cruised at {speed_str} for {duration_str}",
            PhaseType.DECELERATION: f"Decelerated for {duration_str}",
            PhaseType.STOP: f"Stopped after {duration_str}",
            PhaseType.IDLE: f"Remained idle for {duration_str}",
            PhaseType.TURNING: f"Made a turn",
        }
        
        return descriptions.get(phase_type, f"Phase: {phase_type.value}")
    
    def _filter_short_phases(self, phases: list[Phase]) -> list[Phase]:
        """Remove phases shorter than minimum duration."""
        return [
            p for p in phases
            if p.duration_seconds >= self.min_phase_duration
        ]
    
    def detect_speed_anomalies(
        self,
        coordinates: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> list[SpeedAnomaly]:
        """
        Detect speed anomalies (spikes, sudden changes).
        
        Args:
            coordinates: List of enriched coordinates
            speed_data: Optional speed data
            
        Returns:
            List of detected speed anomalies
        """
        anomalies: list[SpeedAnomaly] = []
        speeds = self._compute_speeds(coordinates, speed_data)
        
        if len(speeds) < 3:
            return anomalies
        
        # Calculate rolling statistics
        window_size = min(10, len(speeds) // 3)
        if window_size < 3:
            window_size = 3
        
        for i in range(window_size, len(speeds)):
            window = speeds[i - window_size:i]
            avg_speed = np.mean(window)
            std_speed = np.std(window)
            
            current_speed = speeds[i]
            coord = coordinates[i]
            
            # Detect spike
            if avg_speed > 0 and current_speed > avg_speed * self.speed_spike_factor:
                severity = self._classify_anomaly_severity(
                    current_speed - avg_speed, avg_speed
                )
                anomalies.append(SpeedAnomaly(
                    timestamp=coord.timestamp,
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    speed_kmh=current_speed,
                    expected_speed_kmh=avg_speed,
                    anomaly_type="speed_spike",
                    severity=severity,
                ))
            
            # Detect sudden drop
            elif avg_speed > 0 and current_speed < avg_speed * 0.3:
                severity = self._classify_anomaly_severity(
                    avg_speed - current_speed, avg_speed
                )
                anomalies.append(SpeedAnomaly(
                    timestamp=coord.timestamp,
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    speed_kmh=current_speed,
                    expected_speed_kmh=avg_speed,
                    anomaly_type="sudden_drop",
                    severity=severity,
                ))
        
        logger.info(f"Detected {len(anomalies)} speed anomalies")
        return anomalies
    
    def _classify_anomaly_severity(
        self,
        deviation: float,
        baseline: float
    ) -> EventSeverity:
        """Classify anomaly severity based on deviation."""
        if baseline <= 0:
            return EventSeverity.MINOR
        
        ratio = abs(deviation) / baseline
        
        if ratio >= 1.0:
            return EventSeverity.CRITICAL
        elif ratio >= 0.7:
            return EventSeverity.SEVERE
        elif ratio >= 0.4:
            return EventSeverity.MODERATE
        else:
            return EventSeverity.MINOR
    
    def detect_idle_periods(
        self,
        coordinates: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> list[tuple[datetime, datetime, float]]:
        """
        Detect extended idle periods.
        
        Args:
            coordinates: List of enriched coordinates
            speed_data: Optional speed data
            
        Returns:
            List of (start_time, end_time, duration_seconds) tuples
        """
        speeds = self._compute_speeds(coordinates, speed_data)
        idle_periods: list[tuple[datetime, datetime, float]] = []
        
        idle_start: datetime | None = None
        
        for i, speed in enumerate(speeds):
            if speed <= self.idle_threshold:
                if idle_start is None:
                    idle_start = coordinates[i].timestamp
            else:
                if idle_start is not None:
                    idle_end = coordinates[i - 1].timestamp
                    duration = (idle_end - idle_start).total_seconds()
                    
                    if duration >= self.idle_anomaly_threshold:
                        idle_periods.append((idle_start, idle_end, duration))
                    
                    idle_start = None
        
        # Handle case where trip ends while idle
        if idle_start is not None:
            idle_end = coordinates[-1].timestamp
            duration = (idle_end - idle_start).total_seconds()
            if duration >= self.idle_anomaly_threshold:
                idle_periods.append((idle_start, idle_end, duration))
        
        return idle_periods
    
    def compute_trip_statistics(
        self,
        coordinates: list[EnrichedCoordinate],
        speed_data: list[SpeedData] | None = None,
    ) -> dict[str, Any]:
        """
        Compute overall trip statistics.
        
        Args:
            coordinates: List of enriched coordinates
            speed_data: Optional speed data
            
        Returns:
            Dictionary with trip statistics
        """
        if not coordinates:
            return {}
        
        speeds = self._compute_speeds(coordinates, speed_data)
        speeds_array = np.array(speeds)
        
        # Filter out idle (for speed statistics)
        moving_speeds = speeds_array[speeds_array > self.idle_threshold]
        
        # Time calculations
        start_time = coordinates[0].timestamp
        end_time = coordinates[-1].timestamp
        total_duration = (end_time - start_time).total_seconds()
        
        # Idle time
        idle_periods = self.detect_idle_periods(coordinates, speed_data)
        total_idle = sum(p[2] for p in idle_periods)
        
        # Moving time
        moving_time = total_duration - total_idle
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_duration_seconds": total_duration,
            "moving_time_seconds": moving_time,
            "idle_time_seconds": total_idle,
            "average_speed_kmh": float(np.mean(speeds_array)) if len(speeds_array) > 0 else 0,
            "average_moving_speed_kmh": float(np.mean(moving_speeds)) if len(moving_speeds) > 0 else 0,
            "max_speed_kmh": float(np.max(speeds_array)) if len(speeds_array) > 0 else 0,
            "min_speed_kmh": float(np.min(speeds_array)) if len(speeds_array) > 0 else 0,
            "speed_std_dev": float(np.std(moving_speeds)) if len(moving_speeds) > 0 else 0,
            "idle_periods": len(idle_periods),
        }
