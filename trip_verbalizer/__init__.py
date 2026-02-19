"""
Trip Verbalizer: Vehicle Telematics to Natural Language Pipeline

A production-ready Python pipeline that converts vehicle telematics trip JSON
into human-readable English narration using a local LLM via llama.cpp.
"""

__version__ = "1.0.0"
__author__ = "Trip Verbalizer Team"

from .pipeline import TripVerbalizerPipeline
from .models import TripData, SemanticSummary

__all__ = [
    "TripVerbalizerPipeline",
    "TripData",
    "SemanticSummary",
]
