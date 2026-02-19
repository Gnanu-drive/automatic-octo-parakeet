"""
HTTP Client Utilities

Provides configured HTTP clients for corporate environments with:
- SSL certificate handling (certifi bundle)
- Proxy support
- Retry with exponential backoff
- Proper timeouts
"""

import asyncio
import logging
from typing import Any

import httpx

from .config import AppConfig, RetryConfig, get_config


logger = logging.getLogger(__name__)


def create_httpx_client(
    config: AppConfig | None = None,
    timeout: float | None = None,
    **kwargs: Any
) -> httpx.AsyncClient:
    """
    Create an httpx AsyncClient configured for corporate environments.
    
    Args:
        config: Application config (uses global if not provided)
        timeout: Override default timeout
        **kwargs: Additional arguments to pass to httpx.AsyncClient
        
    Returns:
        Configured AsyncClient
    """
    if config is None:
        config = get_config()
    
    # Build client kwargs
    client_kwargs: dict[str, Any] = {
        "verify": config.ssl.get_verify_path(),
        "timeout": httpx.Timeout(timeout or 30.0, connect=10.0),
        "follow_redirects": True,
    }
    
    # Add proxy if configured - use mounts for per-protocol proxies
    if config.proxy.is_configured:
        proxy_dict = config.proxy.get_proxy_dict()
        if proxy_dict:
            # httpx 0.25+ uses mounts for per-protocol proxies
            # Use AsyncHTTPTransport for async clients
            mounts = {}
            for protocol, proxy_url in proxy_dict.items():
                mounts[protocol] = httpx.AsyncHTTPTransport(proxy=proxy_url)
            client_kwargs["mounts"] = mounts
    
    # Merge with user kwargs
    client_kwargs.update(kwargs)
    
    return httpx.AsyncClient(**client_kwargs)


def create_sync_httpx_client(
    config: AppConfig | None = None,
    timeout: float | None = None,
    **kwargs: Any
) -> httpx.Client:
    """
    Create a synchronous httpx Client configured for corporate environments.
    
    Args:
        config: Application config (uses global if not provided)
        timeout: Override default timeout
        **kwargs: Additional arguments to pass to httpx.Client
        
    Returns:
        Configured Client
    """
    if config is None:
        config = get_config()
    
    client_kwargs: dict[str, Any] = {
        "verify": config.ssl.get_verify_path(),
        "timeout": httpx.Timeout(timeout or 30.0, connect=10.0),
        "follow_redirects": True,
    }
    
    # Add proxy if configured - use mounts for per-protocol proxies
    if config.proxy.is_configured:
        proxy_dict = config.proxy.get_proxy_dict()
        if proxy_dict:
            # httpx 0.25+ uses mounts for per-protocol proxies
            mounts = {}
            for protocol, proxy_url in proxy_dict.items():
                mounts[protocol] = httpx.HTTPTransport(proxy=proxy_url)
            client_kwargs["mounts"] = mounts

    client_kwargs.update(kwargs)
    
    return httpx.Client(**client_kwargs)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    retry_config: RetryConfig | None = None,
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs: Any
) -> httpx.Response:
    """
    Make HTTP request with retry and exponential backoff.
    
    Args:
        client: httpx AsyncClient
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        retry_config: Retry configuration
        retry_on_status: Status codes to retry on
        **kwargs: Additional request arguments
        
    Returns:
        HTTP Response
        
    Raises:
        httpx.HTTPError: If all retries fail
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    last_exception: Exception | None = None
    
    for attempt in range(retry_config.max_attempts):
        try:
            response = await client.request(method, url, **kwargs)
            
            # Check if we should retry based on status
            if response.status_code in retry_on_status:
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"Request to {url} returned {response.status_code}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{retry_config.max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue
            
            return response
            
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exception = e
            
            if attempt < retry_config.max_attempts - 1:
                delay = retry_config.get_delay(attempt)
                logger.warning(
                    f"Request to {url} failed: {e}, "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{retry_config.max_attempts})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Request to {url} failed after {retry_config.max_attempts} attempts")
    
    if last_exception:
        raise last_exception
    
    # This shouldn't happen, but just in case
    raise httpx.RequestError(f"Request failed after {retry_config.max_attempts} attempts")


class RateLimiter:
    """Simple async rate limiter."""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until next request is allowed."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self._last_request_time = asyncio.get_event_loop().time()
    
    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
