"""
Geo Enrichment Module (Corporate Environment Ready)

This module handles reverse geocoding of coordinates using multiple services
with fallback support, caching, rate limiting, SSL handling, and proxy support.

Features:
- SSL certificate handling (certifi bundle)
- Proxy support (HTTP_PROXY/HTTPS_PROXY)
- Retry with exponential backoff
- Rate limiting (1 req/sec for Nominatim)
- SQLite caching layer
- Graceful fallbacks between services
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from .config import AppConfig, RetryConfig, get_config
from .http_utils import create_httpx_client, request_with_retry, RateLimiter
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
        logger.info(f"📦 Geocode cache initialized at {self.db_path}")
    
    @staticmethod
    def _make_key(lat: float, lon: float, precision: int = 5) -> str:
        """Generate cache key from coordinates."""
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
        
        try:
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
                        logger.debug(f"✅ Cache hit for ({lat}, {lon})")
                        return GeoLocation.model_validate_json(row[0])
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
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
        
        try:
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
            logger.debug(f"💾 Cached geocode result for ({lat}, {lon}) from {source}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def cleanup(self) -> int:
        """Remove expired entries and enforce max size."""
        if not self._initialized:
            await self.initialize()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=self.ttl_days)).isoformat()
        deleted = 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM geocode_cache WHERE cached_at < ?",
                    (cutoff_date,)
                )
                deleted += cursor.rowcount
                
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
            
            logger.info(f"🧹 Cache cleanup: removed {deleted} entries")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
        
        return deleted


class NominatimGeocoder:
    """
    Nominatim reverse geocoding service with corporate environment support.
    
    Features:
    - SSL certificate verification (certifi)
    - Proxy support
    - Rate limiting (1 req/sec)
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        base_url: str = "https://nominatim.openstreetmap.org",
        user_agent: str = "trip-verbalizer/1.0 (corporate-env)",
        timeout: float = 15.0,
        rate_limit: float = 1.0,
        retry_config: RetryConfig | None = None,
        app_config: AppConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.rate_limiter = RateLimiter(1.0 / rate_limit)
        self.retry_config = retry_config or RetryConfig(max_attempts=3, base_delay=1.0)
        self.app_config = app_config or get_config()
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        client: httpx.AsyncClient
    ) -> GeoLocation:
        """
        Perform reverse geocoding with rate limiting and retries.
        
        Args:
            lat: Latitude
            lon: Longitude
            client: Async HTTP client (already configured with SSL/proxy)
            
        Returns:
            GeoLocation with enriched data
            
        Raises:
            GeocodingError: If geocoding fails after retries
        """
        # Rate limit
        async with self.rate_limiter:
            last_error: Exception | None = None
            
            for attempt in range(self.retry_config.max_attempts):
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
                        headers={
                            "User-Agent": self.user_agent,
                            "Accept": "application/json",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if "error" in data:
                        raise GeocodingError(f"Nominatim API error: {data['error']}")
                    
                    logger.debug(f"✅ Nominatim success for ({lat}, {lon})")
                    return self._parse_response(lat, lon, data)
                    
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 429:  # Rate limited
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"⏳ Nominatim rate limited, waiting {delay:.1f}s "
                            f"(attempt {attempt + 1}/{self.retry_config.max_attempts})"
                        )
                        await asyncio.sleep(delay)
                    elif e.response.status_code >= 500:  # Server error
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"🔄 Nominatim server error {e.response.status_code}, "
                            f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise GeocodingError(f"Nominatim HTTP error: {e}")
                        
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    last_error = e
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"🔄 Nominatim connection error, retrying in {delay:.1f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    raise GeocodingError(f"Nominatim geocoding failed: {e}")
            
            raise GeocodingError(
                f"Nominatim failed after {self.retry_config.max_attempts} attempts: {last_error}"
            )
    
    def _parse_response(
        self,
        lat: float,
        lon: float,
        data: dict[str, Any]
    ) -> GeoLocation:
        """Parse Nominatim response into GeoLocation."""
        address = data.get("address", {})
        
        road_name = (
            address.get("road") or
            address.get("pedestrian") or
            address.get("footway") or
            address.get("path") or
            address.get("highway")
        )
        
        locality = (
            address.get("neighbourhood") or
            address.get("suburb") or
            address.get("quarter") or
            address.get("hamlet")
        )
        
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
    Photon reverse geocoding service (Komoot) with corporate environment support.
    
    Features:
    - Proper headers to avoid 403 errors
    - SSL certificate verification
    - Proxy support
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        base_url: str = "https://photon.komoot.io",
        user_agent: str = "trip-verbalizer/1.0 (corporate-env)",
        timeout: float = 15.0,
        retry_config: RetryConfig | None = None,
        app_config: AppConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig(max_attempts=3, base_delay=1.0)
        self.app_config = app_config or get_config()
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        client: httpx.AsyncClient
    ) -> GeoLocation:
        """
        Perform reverse geocoding with retries.
        
        Args:
            lat: Latitude
            lon: Longitude
            client: Async HTTP client
            
        Returns:
            GeoLocation with enriched data
        """
        last_error: Exception | None = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Add required headers to avoid 403
                response = await client.get(
                    f"{self.base_url}/reverse",
                    params={
                        "lat": lat,
                        "lon": lon,
                    },
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                
                features = data.get("features", [])
                if not features:
                    raise GeocodingError("No results from Photon")
                
                logger.debug(f"✅ Photon success for ({lat}, {lon})")
                return self._parse_response(lat, lon, features[0])
                
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                
                if status == 403:
                    # Photon may block requests - don't retry, just fail
                    logger.warning(
                        f"🚫 Photon returned 403 Forbidden - service may be blocked "
                        f"by corporate firewall or rate limiting"
                    )
                    raise GeocodingError(f"Photon access forbidden (403)")
                    
                elif status == 429 or status >= 500:
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"🔄 Photon error {status}, retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                else:
                    raise GeocodingError(f"Photon HTTP error: {e}")
                    
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"🔄 Photon connection error, retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                if "No results" in str(e):
                    raise
                raise GeocodingError(f"Photon geocoding failed: {e}")
        
        raise GeocodingError(
            f"Photon failed after {self.retry_config.max_attempts} attempts: {last_error}"
        )
    
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
    Main geo enrichment orchestrator for corporate environments.
    
    Features:
    - SSL certificate handling (certifi bundle)
    - Proxy support (HTTP_PROXY/HTTPS_PROXY)
    - SQLite caching layer
    - Rate limiting for Nominatim
    - Graceful fallback: Nominatim -> Photon -> minimal location
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache: GeoCache | None = None,
        app_config: AppConfig | None = None,
    ):
        """
        Initialize the geo enricher.
        
        Args:
            config: Configuration dictionary (from config.yaml)
            cache: Optional cache instance (creates default if None)
            app_config: Application config for SSL/proxy settings
        """
        self.config = config or {}
        self.app_config = app_config or get_config()
        geo_config = self.config.get("geocoding", {})
        
        # Log configuration
        self._log_configuration()
        
        # Initialize retry config
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
        )
        
        # Initialize geocoders with corporate settings
        nominatim_config = geo_config.get("nominatim", {})
        self.nominatim = NominatimGeocoder(
            base_url=nominatim_config.get("base_url", self.app_config.geocoding.nominatim_url),
            user_agent=self.app_config.geocoding.user_agent,
            timeout=self.app_config.geocoding.timeout,
            rate_limit=nominatim_config.get("rate_limit_delay", 1.0),
            retry_config=retry_config,
            app_config=self.app_config,
        )
        
        photon_config = geo_config.get("photon", {})
        self.photon = PhotonGeocoder(
            base_url=photon_config.get("base_url", self.app_config.geocoding.photon_url),
            user_agent=self.app_config.geocoding.user_agent,
            timeout=self.app_config.geocoding.timeout,
            retry_config=retry_config,
            app_config=self.app_config,
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
        self._photon_blocked = False  # Track if Photon is blocked
    
    def _log_configuration(self) -> None:
        """Log geocoding configuration."""
        ssl_mode = "INSECURE" if self.app_config.ssl.insecure_ssl else "certifi"
        proxy_status = "configured" if self.app_config.proxy.is_configured else "not configured"
        
        logger.info("=" * 50)
        logger.info("🌍 Geocoding Configuration")
        logger.info(f"   SSL Mode: {ssl_mode}")
        logger.info(f"   Proxy: {proxy_status}")
        logger.info(f"   Nominatim: {self.app_config.geocoding.nominatim_url}")
        logger.info(f"   Photon: {self.app_config.geocoding.photon_url}")
        logger.info("=" * 50)
    
    async def __aenter__(self) -> "GeoEnricher":
        """Async context manager entry - create configured HTTP client."""
        self._client = create_httpx_client(
            config=self.app_config,
            timeout=self.app_config.geocoding.timeout,
        )
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
        Reverse geocode a single coordinate with fallback chain.
        
        Order: Cache -> Nominatim -> Photon -> Minimal location
        
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
            self._client = create_httpx_client(
                config=self.app_config,
                timeout=self.app_config.geocoding.timeout,
            )
        
        # Try Nominatim first
        try:
            result = await self.nominatim.reverse_geocode(lat, lon, self._client)
            if self.cache:
                await self.cache.set(lat, lon, result, source="nominatim")
            return result
        except GeocodingError as e:
            logger.warning(f"⚠️  Nominatim failed for ({lat}, {lon}): {e}")
        
        # Fallback to Photon (if not blocked)
        if not self._photon_blocked:
            try:
                result = await self.photon.reverse_geocode(lat, lon, self._client)
                if self.cache:
                    await self.cache.set(lat, lon, result, source="photon")
                return result
            except GeocodingError as e:
                if "403" in str(e) or "forbidden" in str(e).lower():
                    self._photon_blocked = True
                    logger.warning(
                        "🚫 Photon blocked (403), falling back to Nominatim only for remaining requests"
                    )
                else:
                    logger.warning(f"⚠️  Photon fallback failed for ({lat}, {lon}): {e}")
        
        # Return minimal location if all services fail
        logger.warning(
            f"⚠️  All geocoding services failed for ({lat}, {lon}), using coordinates only"
        )
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
        geocoded_count = 0
        cached_count = 0
        
        for i, coord in enumerate(coordinates):
            enriched_coord = EnrichedCoordinate.from_coordinate(coord)
            
            if i % sample_rate == 0:
                try:
                    # Check if we got a cache hit
                    was_cached = False
                    if self.cache:
                        cached = await self.cache.get(coord.latitude, coord.longitude)
                        if cached:
                            was_cached = True
                            cached_count += 1
                    
                    location = await self.reverse_geocode(
                        coord.latitude,
                        coord.longitude
                    )
                    enriched_coord.location = location
                    
                    if not was_cached:
                        geocoded_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to geocode point {i}: {e}")
            
            enriched.append(enriched_coord)
        
        # Interpolate missing geocodes
        self._interpolate_locations(enriched)
        
        logger.info(
            f"📍 Enriched {len(enriched)} coordinates "
            f"(geocoded: {geocoded_count}, cached: {cached_count})"
        )
        return enriched
    
    def _interpolate_locations(
        self,
        coordinates: list[EnrichedCoordinate]
    ) -> None:
        """Fill in missing locations by interpolating from nearby geocoded points."""
        last_location: GeoLocation | None = None
        
        for coord in coordinates:
            if coord.location:
                last_location = coord.location
            elif last_location:
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
