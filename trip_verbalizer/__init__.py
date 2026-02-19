"""
Trip Verbalizer: Vehicle Telematics to Natural Language Pipeline

A production-ready Python pipeline that converts vehicle telematics trip JSON
into human-readable English narration using a local LLM via llama.cpp.

Supports two generation modes:
- narrative: Natural storytelling narration (default)
- navigation_past: Third-person past-tense procedural driving report
"""

__version__ = "1.0.0"
__author__ = "Trip Verbalizer Team"

from .pipeline import TripVerbalizerPipeline
from .models import (
    GenerationMode,
    NavigationAction,
    NavigationInstruction,
    SemanticSummary,
    TripData,
)

__all__ = [
    "GenerationMode",
    "NavigationAction",
    "NavigationInstruction",
    "SemanticSummary",
    "TripData",
    "TripVerbalizerPipeline",
]
