#!/usr/bin/env python3
"""
Trip Verbalizer CLI

Command-line interface for the vehicle telematics trip verbalization pipeline.

Usage:
    python -m trip_verbalizer.main <trip.json> [options]
    trip-verbalizer <trip.json> [options]

Examples:
    # Basic usage
    python -m trip_verbalizer.main sample_trip.json

    # With custom config
    python -m trip_verbalizer.main sample_trip.json --config custom_config.yaml

    # Save output to file
    python -m trip_verbalizer.main sample_trip.json --output narration.txt

    # Use mock LLM for testing
    python -m trip_verbalizer.main sample_trip.json --mock-llm

    # Include debug output
    python -m trip_verbalizer.main sample_trip.json --debug
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .models import TripData, NarrationOutput
from .pipeline import TripVerbalizerPipeline, PipelineError
from .utils.helpers import load_config, setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="trip-verbalizer",
        description="Convert vehicle telematics trip JSON into natural language narration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sample_trip.json
  %(prog)s sample_trip.json --config config.yaml
  %(prog)s sample_trip.json --output narration.txt --json-output metadata.json
  %(prog)s sample_trip.json --mock-llm --debug
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "trip_file",
        type=str,
        help="Path to trip JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for narration text"
    )
    
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output file for full JSON result with metadata"
    )
    
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Path to markdown file with additional context/anomaly notes"
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM client (for testing without LLM server)"
    )
    
    parser.add_argument(
        "--geocode-rate",
        type=int,
        default=5,
        help="Geocode every Nth point (default: 5, use 1 for all)"
    )
    
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Include debug logs in output"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging to console"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all console output except errors"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser


def load_trip_file(file_path: str) -> TripData:
    """Load and validate trip JSON file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Trip file not found: {file_path}")
    
    if not path.suffix.lower() == ".json":
        raise ValueError(f"Expected JSON file, got: {path.suffix}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return TripData.model_validate(data)


def load_context_file(file_path: str | None) -> str | None:
    """Load additional context markdown file."""
    if not file_path:
        return None
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {file_path}")
    
    with open(path, "r") as f:
        return f.read()


def save_output(
    result: NarrationOutput,
    output_file: str | None,
    json_output: str | None,
    include_debug: bool = False
) -> None:
    """Save output to files."""
    # Save narration text
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            f.write(result.narration)
            f.write("\n")
    
    # Save full JSON result
    if json_output:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build output dict
        output_dict: dict[str, Any] = {
            "narration": result.narration,
            "generated_at": result.generated_at.isoformat(),
            "model_used": result.model_used,
            "processing_time_seconds": result.processing_time_seconds,
            "trip_summary": result.semantic_summary.trip_summary.model_dump(),
            "route_description": result.semantic_summary.route_description.model_dump(),
            "driver_behavior": result.semantic_summary.driver_behavior.model_dump(),
            "phase_count": len(result.semantic_summary.phases),
            "event_count": len(result.semantic_summary.events),
            "turn_count": len(result.semantic_summary.turns),
        }
        
        if include_debug:
            output_dict["debug_logs"] = result.debug_logs
            output_dict["warnings"] = result.warnings
        
        with open(path, "w") as f:
            json.dump(output_dict, f, indent=2, default=str)


def print_result_rich(result: NarrationOutput, include_debug: bool = False) -> None:
    """Print result using rich formatting."""
    console = Console()
    
    # Print narration
    console.print(Panel(
        result.narration,
        title="[bold green]Trip Narration[/bold green]",
        border_style="green"
    ))
    
    # Print summary table
    table = Table(title="Trip Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    summary = result.semantic_summary.trip_summary
    table.add_row("Trip ID", summary.trip_id)
    table.add_row("Duration", summary.duration)
    table.add_row("Distance", summary.distance)
    table.add_row("Start", summary.start_location)
    table.add_row("End", summary.end_location)
    table.add_row("Avg Speed", summary.average_speed)
    table.add_row("Max Speed", summary.max_speed)
    table.add_row("Events", str(summary.event_count))
    
    console.print(table)
    
    # Print behavior rating
    behavior = result.semantic_summary.driver_behavior
    rating_color = "green" if behavior.overall_rating >= 7 else "yellow" if behavior.overall_rating >= 5 else "red"
    console.print(f"\n[bold]Driver Behavior Rating:[/bold] [{rating_color}]{behavior.overall_rating}/10[/{rating_color}]")
    
    # Print processing info
    console.print(f"\n[dim]Generated in {result.processing_time_seconds:.2f}s using {result.model_used}[/dim]")
    
    # Print debug logs if requested
    if include_debug and result.debug_logs:
        console.print("\n[bold yellow]Debug Logs:[/bold yellow]")
        for log in result.debug_logs:
            console.print(f"  [dim]{log}[/dim]")
    
    # Print warnings
    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]⚠ {warning}[/yellow]")


def print_result_simple(result: NarrationOutput, include_debug: bool = False) -> None:
    """Print result using simple formatting."""
    print("\n" + "=" * 60)
    print("TRIP NARRATION")
    print("=" * 60)
    print(result.narration)
    
    print("\n" + "-" * 60)
    print("TRIP SUMMARY")
    print("-" * 60)
    
    summary = result.semantic_summary.trip_summary
    print(f"  Trip ID:     {summary.trip_id}")
    print(f"  Duration:    {summary.duration}")
    print(f"  Distance:    {summary.distance}")
    print(f"  Start:       {summary.start_location}")
    print(f"  End:         {summary.end_location}")
    print(f"  Avg Speed:   {summary.average_speed}")
    print(f"  Max Speed:   {summary.max_speed}")
    print(f"  Events:      {summary.event_count}")
    
    behavior = result.semantic_summary.driver_behavior
    print(f"\n  Driver Rating: {behavior.overall_rating}/10")
    
    print(f"\n  Generated in {result.processing_time_seconds:.2f}s using {result.model_used}")
    
    if include_debug and result.debug_logs:
        print("\n" + "-" * 60)
        print("DEBUG LOGS")
        print("-" * 60)
        for log in result.debug_logs:
            print(f"  {log}")
    
    if result.warnings:
        print("\n" + "-" * 60)
        print("WARNINGS")
        print("-" * 60)
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    print()


async def run_pipeline(args: argparse.Namespace) -> NarrationOutput:
    """Run the pipeline with given arguments."""
    # Load trip data
    trip_data = load_trip_file(args.trip_file)
    
    # Load additional context
    additional_context = load_context_file(args.context)
    
    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Create and run pipeline
    pipeline = TripVerbalizerPipeline(
        config=config,
        use_mock_llm=args.mock_llm
    )
    
    return await pipeline.process(
        trip_data=trip_data,
        additional_context=additional_context,
        geocode_sample_rate=args.geocode_rate
    )


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "WARNING"
    
    setup_logging(level=log_level)
    
    try:
        # Run with progress indicator if rich is available
        if RICH_AVAILABLE and not args.quiet:
            console = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing trip...", total=None)
                result = asyncio.run(run_pipeline(args))
                progress.update(task, completed=True)
        else:
            result = asyncio.run(run_pipeline(args))
        
        # Save outputs if specified
        save_output(
            result,
            args.output,
            args.json_output,
            include_debug=args.debug
        )
        
        # Print result
        if not args.quiet:
            if RICH_AVAILABLE:
                print_result_rich(result, include_debug=args.debug)
            else:
                print_result_simple(result, include_debug=args.debug)
        
        return 0
        
    except FileNotFoundError as e:
        if RICH_AVAILABLE:
            Console().print(f"[red]Error:[/red] {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except PipelineError as e:
        if RICH_AVAILABLE:
            Console().print(f"[red]Pipeline Error:[/red] {e}")
        else:
            print(f"Pipeline Error: {e}", file=sys.stderr)
        return 2
        
    except Exception as e:
        if RICH_AVAILABLE:
            Console().print(f"[red]Unexpected Error:[/red] {e}")
        else:
            print(f"Unexpected Error: {e}", file=sys.stderr)
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        return 3


if __name__ == "__main__":
    sys.exit(main())
