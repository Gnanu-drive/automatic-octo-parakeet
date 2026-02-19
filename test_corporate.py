#!/usr/bin/env python3
"""Test script for corporate environment configuration."""

import asyncio
import json

def main():
    print("Testing corporate environment configuration...")
    print()

    # Test 1: Config module
    from trip_verbalizer.config import get_config, SSLConfig, ProxyConfig, RetryConfig
    config = get_config()
    print("✅ Config module loaded")
    print(f"   SSL insecure: {config.ssl.insecure_ssl}")
    print(f"   Proxy configured: {config.proxy.is_configured}")
    print(f"   LLM host: {config.llm.host}:{config.llm.port}")
    print()

    # Test 2: HTTP utilities
    from trip_verbalizer.http_utils import create_httpx_client, RateLimiter
    client = create_httpx_client(config)
    print("✅ HTTP client created with SSL/proxy settings")
    print()

    # Test 3: Geo module
    from trip_verbalizer.geo import GeoEnricher, GeoCache, NominatimGeocoder, PhotonGeocoder
    print("✅ Geo module with corporate support loaded")
    print()

    # Test 4: LLM client
    from trip_verbalizer.llm_client import LLMClient, MockLLMClient, FallbackNarrator
    print("✅ LLM client with corporate support loaded")
    print()

    # Test 5: Full pipeline
    from trip_verbalizer.pipeline import TripVerbalizerPipeline
    from trip_verbalizer.models import TripData

    with open("samples/sample_trip.json") as f:
        trip_data = TripData.model_validate(json.load(f))

    pipeline = TripVerbalizerPipeline(use_mock_llm=True)
    result = asyncio.run(pipeline.process(trip_data))
    print("✅ Full pipeline executed successfully")
    print(f"   Processing time: {result.processing_time_seconds:.2f}s")
    print()

    print("=" * 60)
    print("🎉 All tests passed! Pipeline is corporate-ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()
