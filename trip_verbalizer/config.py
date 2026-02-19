"""
Central Configuration Module for Trip Verbalizer

Handles environment-specific settings for corporate environments:
- SSL/TLS configuration
- Proxy settings
- API timeouts
- LLM server settings
- Retry policies
"""

import logging
import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import certifi


logger = logging.getLogger(__name__)


@dataclass
class SSLConfig:
    """SSL/TLS configuration for HTTPS requests."""
    
    # Use certifi bundle by default
    use_certifi: bool = True
    
    # Allow insecure SSL (ONLY for development)
    insecure_ssl: bool = False
    
    # Custom CA bundle path
    ca_bundle_path: str | None = None
    
    @classmethod
    def from_env(cls) -> "SSLConfig":
        """Create SSL config from environment variables."""
        insecure = os.getenv("TRIP_VERBALIZER_INSECURE_SSL", "").lower() in ("true", "1", "yes")
        ca_bundle = os.getenv("TRIP_VERBALIZER_CA_BUNDLE")
        
        config = cls(
            insecure_ssl=insecure,
            ca_bundle_path=ca_bundle,
        )
        
        if insecure:
            logger.warning(
                "⚠️  INSECURE SSL MODE ENABLED - Certificate verification disabled. "
                "Use only for development!"
            )
        elif ca_bundle:
            logger.info(f"Using custom CA bundle: {ca_bundle}")
        else:
            logger.info(f"Using certifi SSL bundle: {certifi.where()}")
        
        return config
    
    def get_ssl_context(self) -> ssl.SSLContext | bool:
        """Get SSL context for httpx/aiohttp."""
        if self.insecure_ssl:
            return False  # Disable verification
        
        if self.ca_bundle_path and Path(self.ca_bundle_path).exists():
            ctx = ssl.create_default_context(cafile=self.ca_bundle_path)
            return ctx
        
        # Use certifi bundle
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    
    def get_verify_path(self) -> str | bool:
        """Get verify parameter for httpx."""
        if self.insecure_ssl:
            return False
        
        if self.ca_bundle_path and Path(self.ca_bundle_path).exists():
            return self.ca_bundle_path
        
        return certifi.where()


@dataclass
class ProxyConfig:
    """Proxy configuration for corporate networks."""
    
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None
    
    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Create proxy config from environment variables."""
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        
        config = cls(
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            no_proxy=no_proxy,
        )
        
        if http_proxy or https_proxy:
            logger.info(f"🔗 Using corporate proxy: HTTP={http_proxy}, HTTPS={https_proxy}")
        
        return config
    
    def get_proxy_dict(self) -> dict[str, str] | None:
        """Get proxy dict for httpx."""
        proxies = {}
        
        if self.http_proxy:
            proxies["http://"] = self.http_proxy
        if self.https_proxy:
            proxies["https://"] = self.https_proxy
        
        return proxies if proxies else None
    
    @property
    def is_configured(self) -> bool:
        """Check if any proxy is configured."""
        return bool(self.http_proxy or self.https_proxy)


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""
    
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-indexed)."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class GeocodingConfig:
    """Geocoding service configuration."""
    
    # Nominatim settings
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    nominatim_rate_limit: float = 1.0  # seconds between requests
    
    # Photon settings
    photon_url: str = "https://photon.komoot.io"
    
    # Common settings
    timeout: float = 15.0
    user_agent: str = "trip-verbalizer/1.0 (corporate-env)"
    
    # Retry config
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    # Cache settings
    cache_enabled: bool = True
    cache_path: str = ".cache/geocode_cache.db"
    cache_ttl_days: int = 30
    
    @classmethod
    def from_env(cls) -> "GeocodingConfig":
        """Create geocoding config from environment."""
        return cls(
            nominatim_url=os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org"),
            photon_url=os.getenv("PHOTON_URL", "https://photon.komoot.io"),
            timeout=float(os.getenv("GEOCODING_TIMEOUT", "15.0")),
            user_agent=os.getenv("GEOCODING_USER_AGENT", "trip-verbalizer/1.0 (corporate-env)"),
        )


@dataclass
class LLMConfig:
    """LLM server configuration."""
    
    host: str = "localhost"
    port: int = 8080
    endpoint: str = "/v1/chat/completions"
    
    # Model parameters
    model: str = "local-model"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    # Connection settings
    timeout: float = 120.0
    connect_timeout: float = 10.0
    
    # Retry config
    retry: RetryConfig = field(default_factory=lambda: RetryConfig(max_attempts=3, base_delay=2.0))
    
    # Health check
    health_check_enabled: bool = True
    health_check_timeout: float = 5.0
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLM config from environment variables."""
        host = os.getenv("LLM_HOST", "localhost")
        port = int(os.getenv("LLM_PORT", "8080"))
        
        logger.info(f"🤖 LLM server configured: {host}:{port}")
        
        return cls(
            host=host,
            port=port,
            model=os.getenv("LLM_MODEL", "local-model"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=float(os.getenv("LLM_TIMEOUT", "120.0")),
        )
    
    @property
    def base_url(self) -> str:
        """Get full base URL for LLM server."""
        return f"http://{self.host}:{self.port}"


@dataclass
class AppConfig:
    """Main application configuration."""
    
    ssl: SSLConfig = field(default_factory=SSLConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    geocoding: GeocodingConfig = field(default_factory=GeocodingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create application config from environment variables.
        
        Environment variables:
            TRIP_VERBALIZER_INSECURE_SSL: Disable SSL verification (dev only)
            TRIP_VERBALIZER_CA_BUNDLE: Custom CA bundle path
            HTTP_PROXY / HTTPS_PROXY: Proxy settings
            LLM_HOST / LLM_PORT: LLM server location
            NOMINATIM_URL / PHOTON_URL: Geocoding service URLs
            GEOCODING_TIMEOUT: Timeout for geocoding requests
            LLM_TIMEOUT: Timeout for LLM requests
        """
        config = cls(
            ssl=SSLConfig.from_env(),
            proxy=ProxyConfig.from_env(),
            geocoding=GeocodingConfig.from_env(),
            llm=LLMConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        
        return config
    
    def log_configuration(self) -> None:
        """Log current configuration summary."""
        logger.info("=" * 60)
        logger.info("Trip Verbalizer Configuration")
        logger.info("=" * 60)
        
        # SSL
        if self.ssl.insecure_ssl:
            logger.warning("SSL: ⚠️  INSECURE (verification disabled)")
        else:
            logger.info(f"SSL: ✅ Secure (using {self.ssl.get_verify_path()})")
        
        # Proxy
        if self.proxy.is_configured:
            logger.info(f"Proxy: ✅ Configured")
        else:
            logger.info("Proxy: ❌ Not configured")
        
        # LLM
        logger.info(f"LLM Server: {self.llm.base_url}")
        
        # Geocoding
        logger.info(f"Nominatim: {self.geocoding.nominatim_url}")
        logger.info(f"Photon: {self.geocoding.photon_url}")
        
        logger.info("=" * 60)


# Global config instance (lazy loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get or create global application config."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset global config (useful for testing)."""
    global _config
    _config = None
