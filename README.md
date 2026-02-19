# Trip Verbalizer

A production-ready Python pipeline that converts vehicle telematics trip JSON into human-readable English narration using a local LLM running via llama.cpp.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for geo enrichment, spatial analysis, temporal analysis, and event processing
- **Async Support**: Fully asynchronous pipeline for efficient I/O operations
- **Geo Enrichment**: Reverse geocoding with Nominatim + Photon fallback, intelligent caching, and rate limiting
- **Spatial Analysis**: Bearing/heading computation, turn detection, route geometry analysis
- **Temporal Analysis**: Trip phase segmentation, speed anomaly detection, idle time analysis
- **Event Analysis**: Severity scoring for driving events with contextual interpretation
- **LLM Integration**: Seamless integration with llama.cpp server for natural language generation
- **Rich CLI**: Beautiful command-line interface with progress indicators and formatted output

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/trip-verbalizer.git
cd trip-verbalizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r trip_verbalizer/requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### 1. Start llama.cpp Server

```bash
# Download and run llama.cpp with your preferred model
./llama-server -m your-model.gguf --port 8080
```

### 2. Run the Pipeline

```bash
# Basic usage
python -m trip_verbalizer.main samples/sample_trip.json

# With custom config
python -m trip_verbalizer.main samples/sample_trip.json --config trip_verbalizer/config.yaml

# Save output to files
python -m trip_verbalizer.main samples/sample_trip.json \
    --output narration.txt \
    --json-output metadata.json

# Include additional context
python -m trip_verbalizer.main samples/sample_trip.json \
    --context samples/anomaly_notes.md

# Use mock LLM for testing (no server required)
python -m trip_verbalizer.main samples/sample_trip.json --mock-llm

# Debug mode
python -m trip_verbalizer.main samples/sample_trip.json --debug --verbose
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRIP VERBALIZER PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Data   │───▶│   Geo    │───▶│ Spatial  │───▶│ Temporal │  │
│  │  Loader  │    │ Enricher │    │ Analyzer │    │ Analyzer │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       │         ┌──────────┐    ┌──────────┐         │         │
│       │         │  Event   │◀───│ Semantic │◀────────┘         │
│       │         │ Analyzer │    │ Builder  │                    │
│       │         └──────────┘    └──────────┘                    │
│       │               │               │                         │
│       │               ▼               ▼                         │
│       │         ┌──────────┐    ┌──────────┐                   │
│       │         │  Prompt  │───▶│   LLM    │                   │
│       │         │ Builder  │    │  Client  │                   │
│       │         └──────────┘    └──────────┘                   │
│       │                               │                         │
│       │                               ▼                         │
│       │                         ┌──────────┐                   │
│       └────────────────────────▶│  Output  │                   │
│                                 │ Handler  │                   │
│                                 └──────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Input Format

The pipeline accepts JSON files with the following structure:

```json
{
  "trip_id": "TRIP-001",
  "vehicle_id": "VEH-ABC123",
  "route": [
    {
      "latitude": 34.0522,
      "longitude": -118.2437,
      "timestamp": "2024-03-15T09:00:00Z",
      "altitude": 50.0,
      "accuracy": 5.0
    }
  ],
  "speed_data": [
    {
      "timestamp": "2024-03-15T09:00:00Z",
      "speed_kmh": 45.0,
      "speed_limit": 50.0
    }
  ],
  "events": [
    {
      "event_id": "EVT-001",
      "event_type": "hard_braking",
      "timestamp": "2024-03-15T09:10:00Z",
      "latitude": 34.0550,
      "longitude": -118.2500,
      "severity": "moderate",
      "value": 0.5
    }
  ],
  "vehicle_conditions": [...],
  "driver": {...}
}
```

## Output

The pipeline produces:

### 1. Natural Language Narration
```
The driver started the vehicle at 9:00 AM from Marina del Rey. The vehicle 
moved northwest along Lincoln Boulevard, gradually accelerating to 60 km/h. 
Around 9:10 AM, a sudden hard braking event occurred near Venice. The driver 
continued toward Santa Monica, maintaining steady speed, before reaching the 
destination at 9:40 AM.
```

### 2. Structured Metadata (JSON)
- Trip summary with duration, distance, speeds
- Route description with roads and direction
- Driver behavior analysis with scoring
- Phase breakdown of the journey
- Event analysis with severity ratings

### 3. Debug Logs (Optional)
- Pipeline stage execution times
- Geocoding cache hits/misses
- LLM request details

## Configuration

See `trip_verbalizer/config.yaml` for all configuration options:

- **Geocoding**: Service URLs, rate limits, caching settings
- **Spatial Analysis**: Turn detection thresholds, cardinal precision
- **Temporal Analysis**: Speed thresholds, phase duration minimums
- **Event Analysis**: Severity thresholds and weights
- **LLM**: Model parameters, timeout, retry settings
- **Output**: Formatting options, included fields

## Project Structure

```
trip_verbalizer/
├── __init__.py          # Package initialization
├── main.py              # CLI entry point
├── pipeline.py          # Pipeline orchestrator
├── models.py            # Pydantic data models
├── geo.py               # Geo enrichment module
├── spatial.py           # Spatial analysis module
├── temporal.py          # Temporal analysis module
├── events.py            # Event analysis module
├── llm_client.py        # LLM client for llama.cpp
├── prompt_builder.py    # Prompt construction
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
└── utils/
    ├── __init__.py
    └── helpers.py       # Utility functions

samples/
├── sample_trip.json     # Example trip data
└── anomaly_notes.md     # Example context file
```

## API Usage

```python
import asyncio
from trip_verbalizer import TripVerbalizerPipeline, TripData

# Load trip data
with open("trip.json") as f:
    trip_dict = json.load(f)
trip_data = TripData.model_validate(trip_dict)

# Create pipeline
pipeline = TripVerbalizerPipeline(
    config_path="config.yaml",
    use_mock_llm=False
)

# Process trip
result = asyncio.run(pipeline.process(trip_data))

# Access results
print(result.narration)
print(result.semantic_summary.trip_summary)
print(result.semantic_summary.driver_behavior.overall_rating)
```

## Requirements

- Python 3.10+
- llama.cpp server running locally (or use `--mock-llm` for testing)
- Internet connection for geocoding (results are cached)

## Dependencies

- `httpx` / `aiohttp`: Async HTTP clients
- `pydantic`: Data validation
- `geopy`: Geocoding utilities
- `shapely`: Geometry operations
- `numpy`: Numerical computations
- `aiosqlite`: Async SQLite for caching
- `aiolimiter`: Rate limiting
- `rich`: CLI formatting
- `structlog`: Structured logging

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests