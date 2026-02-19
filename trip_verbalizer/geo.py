"""
Geo Enrichment Module

This module handles reverse geocoding of coordinates using multiple services
with fallback support, caching, and rate limiting.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel

from .models import Coordinate, EnrichedCoordinate, GeoLocation


logger = logging.getLogger(__name__)


class GeocodingError(Exception):
    """Exception raised when geocoding fails."""
    pass


class GeoCache:
    """
    SQLite-based cache for geocoding results.
    
    Stores results with TTL and provides fast lookups by coordinate hash.
    """
    
    def __init__(
        self,
        db_path: str | Path = ".cache/geocode_cache.db",
        ttl_days: int = 30,
        max_entries: int = 10000,
    ):
        self.db_path = Path(db_path)
        self.ttl_days = ttl_days
        self.max_entries = max_entries
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the cache database."""
        if self._initialized:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS geocode_cache (
                    cache_key TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cached_at 
                ON geocode_cache(cached_at)
            """)
            await db.commit()
        
        self._initialized = True
        logger.info(f"Geocode cache initialized at {self.db_path}")
    
    @staticmethod
    def _make_key(lat: float, lon: float, precision: int = 5) -> str:
        """Generate cache key from coordinates."""
        # Round to specified precision for cache hits on nearby points
        rounded_lat = round(lat, precision)
        rounded_lon = round(lon, precision)
        key_str = f"{rounded_lat},{rounded_lon}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, lat: float, lon: float) -> GeoLocation | None:
        """Retrieve cached geocoding result."""
        if not self._initialized:
            await self.initialize()
        
        cache_key = self._make_key(lat, lon)
        cutoff_date = (datetime.utcnow() - timedelta(days=self.ttl_days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT result_json FROM geocode_cache 
                WHERE cache_key = ? AND cached_at > ?
                """,
                (cache_key, cutoff_date)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    logger.debug(f"Cache hit for ({lat}, {lon})")
                    return GeoLocation.model_validate_json(row[0])
        
        return None
    
    async def set(
        self,
        lat: float,
        lon: float,
        result: GeoLocation,
        source: str = "nominatim"
    ) -> None:
        """Store geocoding result in cache."""
        if not self._initialized:
            await self.initialize()
        
        cache_key = self._make_key(lat, lon)
        result_json = result.model_dump_json()
        cached_at = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO geocode_cache 
                (cache_key, latitude, longitude, result_json, source, cached_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, lat, lon, result_json, source, cached_at)
            )
            await db.commit()
        
        logger.debug(f"Cached geocode result for ({lat}, {lon})")
    
    async def cleanup(self) -> int:
        """Remove expired entries and enforce max size."""
        if not self._initialized:
            await self.initialize()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=self.ttl_days)).isoformat()
        deleted = 0
        
        async with aiosqlite.connect(self.db_path) as db:
            # Remove expired entries
            cursor = await db.execute(
                "DELETE FROM geocode_cache WHERE cached_at < ?",
                (cutoff_date,)
            )
            deleted += cursor.rowcount
            
            # Enforce max entries (keep newest)
            await db.execute(
                """
                DELETE FROM geocode_cache 
                WHERE cache_key NOT IN (
                    SELECT cache_key FROM geocode_cache 
                    ORDER BY cached_at DESC 
                    LIMIT ?
                )
                """,
                (self.max_entries,)
            )
            deleted += cursor.rowcount
            
            await db.commit()
        
        logger.info(f"Cache cleanup: removed {deleted} entries")
        return deleted


class NominatimGeocoder:
    """
    Nominatim reverse geocoding service.
    
    Respects rate limits and provides structured location data.
    """
    
    def __init__(
        self,
        base_url: str = "https://nominatim.openstreetmap.org",
        user_agent: str = "trip_verbalizer/1.0",
        timeout: float = 10.0,
        rate_limit: float = 1.0,  # requests per second
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.limiter = AsyncLimiter(1, rate_limit)  # 1 request per rate_limit seconds
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        client: httpx.AsyncClient
    ) -> GeoLocation:
        """
        Perform reverse geocoding for a coordinate.
        
        Args:
            lat: Latitude
            lon: Longitude
            client: Async HTTP client
            
        Returns:
            GeoLocation with enriched data
            
        Raises:
            GeocodingError: If geocoding fails
        """
        async with self.limiter:
            try:
                response = await client.get(
                    f"{self.base_url}/reverse",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "json",
                        "addressdetails": 1,
                        "zoom": 18,
                    },
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    raise GeocodingError(f"Nominatim error: {data['error']}")
                
                return self._parse_response(lat, lon, data)
                
            except httpx.HTTPError as e:
                raise GeocodingError(f"HTTP error during geocoding: {e}")
            except Exception as e:
                raise GeocodingError(f"Geocoding failed: {e}")
    
    def _parse_response(
        self,
        lat: float,
        lon: float,
        data: dict[str, Any]
    ) -> GeoLocation:
        """Parse Nominatim response into GeoLocation."""
        address = data.get("address", {})
        
        # Extract road name (try multiple fields)
        road_name = (
            address.get("road") or
            address.get("pedestrian") or
            address.get("footway") or
            address.get("path") or
            address.get("highway")
        )
        
        # Extract locality
        locality = (
            address.get("neighbourhood") or
            address.get("suburb") or
            address.get("quarter") or
            address.get("hamlet")
        )
        
        # Extract city
        city = (
            address.get("city") or
            address.get("town") or
            address.get("village") or
            address.get("municipality")
        )
        
        return GeoLocation(
            latitude=lat,
            longitude=lon,
            timestamp=datetime.utcnow(),
            road_name=road_name,
            locality=locality,
            city=city,
            state=address.get("state"),
            country=address.get("country"),
            postal_code=address.get("postcode"),
            display_name=data.get("display_name"),
        )


class PhotonGeocoder:
    """
    Photon reverse geocoding service (Komoot).
    
    Used as fallback when Nominatim is unavailable.
    """
    
    def __init__(
        self,
        base_url: str = "https://photon.komoot.io",
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        client: httpx.AsyncClient
    ) -> GeoLocation:
        """
        Perform reverse geocoding using Photon.
        
        Args:
            lat: Latitude
            lon: Longitude
            client: Async HTTP client
            
        Returns:
            GeoLocation with enriched data
        """
        try:
            response = await client.get(
                f"{self.base_url}/reverse",
                params={
                    "lat": lat,
                    "lon": lon,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            features = data.get("features", [])
            if not features:
                raise GeocodingError("No results from Photon")
            
            return self._parse_response(lat, lon, features[0])
            
        except httpx.HTTPError as e:
            raise GeocodingError(f"Photon HTTP error: {e}")
        except Exception as e:
            raise GeocodingError(f"Photon geocoding failed: {e}")
    
    def _parse_response(
        self,
        lat: float,
        lon: float,
        feature: dict[str, Any]
    ) -> GeoLocation:
        """Parse Photon response into GeoLocation."""
        props = feature.get("properties", {})
        
        return GeoLocation(
            latitude=lat,
            longitude=lon,
            timestamp=datetime.utcnow(),
            road_name=props.get("street"),
            locality=props.get("locality") or props.get("district"),
            city=props.get("city"),
            state=props.get("state"),
            country=props.get("country"),
            postal_code=props.get("postcode"),
            display_name=props.get("name"),
        )


class GeoEnricher:
    """
    Main geo enrichment orchestrator.
    
    Handles batch geocoding with caching, rate limiting, and fallback services.
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache: GeoCache | None = None,
    ):
        """
        Initialize the geo enricher.
        
        Args:
            config: Configuration dictionary (from config.yaml)
            cache: Optional cache instance (creates default if None)
        """
        self.config = config or {}
        geo_config = self.config.get("geocoding", {})
        
        # Initialize geocoders
        nominatim_config = geo_config.get("nominatim", {})
        self.nominatim = NominatimGeocoder(
            base_url=nominatim_config.get("base_url", "https://nominatim.openstreetmap.org"),
            user_agent=nominatim_config.get("user_agent", "trip_verbalizer/1.0"),
            timeout=nominatim_config.get("timeout", 10.0),
            rate_limit=nominatim_config.get("rate_limit_delay", 1.0),
        )
        
        photon_config = geo_config.get("photon", {})
        self.photon = PhotonGeocoder(
            base_url=photon_config.get("base_url", "https://photon.komoot.io"),
            timeout=photon_config.get("timeout", 10.0),
        )
        
        # Initialize cache
        cache_config = geo_config.get("cache", {})
        if cache:
            self.cache = cache
        elif cache_config.get("enabled", True):
            self.cache = GeoCache(
                db_path=cache_config.get("sqlite_path", ".cache/geocode_cache.db"),
                ttl_days=cache_config.get("ttl_days", 30),
                max_entries=cache_config.get("max_entries", 10000),
            )
        else:
            self.cache = None
        
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "GeoEnricher":
        """Async context manager entry."""
        self._client = httpx.AsyncClient()
        if self.cache:
            await self.cache.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
    ) -> GeoLocation:
        """
        Reverse geocode a single coordinate.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            GeoLocation with place information
        """
        # Check cache first
        if self.cache:
            cached = await self.cache.get(lat, lon)
            if cached:
                return cached
        
        # Ensure client is available
        if not self._client:
            self._client = httpx.AsyncClient()
        
        # Try Nominatim first
        try:
            result = await self.nominatim.reverse_geocode(lat, lon, self._client)
            if self.cache:
                await self.cache.set(lat, lon, result, source="nominatim")
            return result
        except GeocodingError as e:
            logger.warning(f"Nominatim failed for ({lat}, {lon}): {e}")
        
        # Fallback to Photon
        try:
            result = await self.photon.reverse_geocode(lat, lon, self._client)
            if self.cache:
                await self.cache.set(lat, lon, result, source="photon")
            return result
        except GeocodingError as e:
            logger.warning(f"Photon fallback failed for ({lat}, {lon}): {e}")
        
        # Return minimal location if all services fail
        return GeoLocation(
            latitude=lat,
            longitude=lon,
            timestamp=datetime.utcnow(),
        )
    
    async def enrich_coordinates(
        self,
        coordinates: list[Coordinate],
        sample_rate: int = 1,
    ) -> list[EnrichedCoordinate]:
        """
        Enrich a list of coordinates with geocoding data.
        
        Args:
            coordinates: List of coordinates to enrich
            sample_rate: Geocode every Nth point (1 = all points)
            
        Returns:
            List of enriched coordinates
        """
        enriched: list[EnrichedCoordinate] = []
        
        for i, coord in enumerate(coordinates):
            # Create enriched coordinate
            enriched_coord = EnrichedCoordinate.from_coordinate(coord)
            
            # Geocode at sample rate
            if i % sample_rate == 0:
                try:
                    location = await self.reverse_geocode(
                        coord.latitude,
                        coord.longitude
                    )
                    enriched_coord.location = location
                except Exception as e:
                    logger.error(f"Failed to geocode point {i}: {e}")
            
            enriched.append(enriched_coord)
        
        # Interpolate missing geocodes
        self._interpolate_locations(enriched)
        
        logger.info(f"Enriched {len(enriched)} coordinates")
        return enriched
    
    def _interpolate_locations(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> None:
        """
        Fill in missing locations by interpolating from nearby geocoded points.
        
        Modifies the list in place.
        """
        last_location: GeoLocation | None = None
        
        for coord in coordinates:
            if coord.location:
                last_location = coord.location
            elif last_location:
                # Copy location info from last known point
                coord.location = GeoLocation(
                    latitude=coord.latitude,
                    longitude=coord.longitude,
                    timestamp=coord.timestamp,
                    road_name=last_location.road_name,
                    locality=last_location.locality,
                    city=last_location.city,
                    state=last_location.state,
                    country=last_location.country,
                )
    
    async def get_key_locations(
        self,
        coordinates: list[Coordinate],
        indices: list[int] | None = None,
    ) -> dict[int, GeoLocation]:
        """
        Geocode specific key points (start, end, events, etc.).
        
        Args:
            coordinates: Full list of coordinates
            indices: Specific indices to geocode (default: first and last)
            
        Returns:
            Dictionary mapping index to GeoLocation
        """
        if indices is None:
            indices = [0, len(coordinates) - 1]
        
        results: dict[int, GeoLocation] = {}
        
        for idx in indices:
            if 0 <= idx < len(coordinates):
                coord = coordinates[idx]
                location = await self.reverse_geocode(
                    coord.latitude,
                    coord.longitude
                )
                results[idx] = location
        
        return results
