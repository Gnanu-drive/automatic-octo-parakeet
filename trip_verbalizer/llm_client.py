"""
LLM Client Module (Corporate Environment Ready)

This module handles communication with the local llama.cpp server
for generating natural language narrations.

Features:
- SSL certificate handling (certifi bundle)
- Proxy support (HTTP_PROXY/HTTPS_PROXY)
- Retry with exponential backoff
- Health check with detailed diagnostics
- Environment variable configuration (LLM_HOST, LLM_PORT)
"""

import asyncio
import logging
import os
from typing import Any

import httpx

from .config import AppConfig, RetryConfig, get_config
from .http_utils import create_httpx_client


logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised when LLM request fails."""
    pass


class LLMConnectionError(LLMError):
    """Exception raised when LLM server is not reachable."""
    pass


class LLMClient:
    """
    Async client for llama.cpp HTTP API with corporate environment support.
    
    Features:
    - SSL certificate handling (uses certifi or insecure mode)
    - Proxy support (reads HTTP_PROXY/HTTPS_PROXY)
    - Environment variables: LLM_HOST, LLM_PORT
    - Retry with exponential backoff
    - Detailed health check diagnostics
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        endpoint: str = "/v1/chat/completions",
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        timeout: float = 120.0,
        retry_config: RetryConfig | None = None,
        app_config: AppConfig | None = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: llama.cpp server base URL (or use LLM_HOST/LLM_PORT env vars)
            endpoint: Completion endpoint path
            model: Model name (for API compatibility)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            retry_config: Retry configuration
            app_config: Application config for SSL/proxy settings
        """
        self.app_config = app_config or get_config()
        
        # Get base URL from parameter, config, or environment
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            host = self.app_config.llm.host
            port = self.app_config.llm.port
            self.base_url = f"http://{host}:{port}"
        
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig(
            max_attempts=self.app_config.llm.retry_attempts,
            base_delay=self.app_config.llm.retry_delay,
        )
        
        self._client: httpx.AsyncClient | None = None
        self._health_checked = False
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self) -> None:
        """Log LLM client configuration."""
        ssl_mode = "INSECURE" if self.app_config.ssl.insecure_ssl else "certifi"
        proxy_status = "configured" if self.app_config.proxy.is_configured else "not configured"
        
        logger.info("=" * 50)
        logger.info("🤖 LLM Client Configuration")
        logger.info(f"   Server: {self.base_url}")
        logger.info(f"   Endpoint: {self.endpoint}")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   SSL Mode: {ssl_mode}")
        logger.info(f"   Proxy: {proxy_status}")
        logger.info(f"   Timeout: {self.timeout}s")
        logger.info(f"   Retries: {self.retry_config.max_attempts}")
        logger.info("=" * 50)
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMClient":
        """
        Create LLM client from configuration dictionary.
        
        Args:
            config: Configuration dict with 'llm' section
            
        Returns:
            Configured LLMClient instance
        """
        llm_config = config.get("llm", {})
        app_config = get_config()
        
        # Build base URL from config or environment
        base_url = llm_config.get("base_url")
        if not base_url:
            host = app_config.llm.host
            port = app_config.llm.port
            base_url = f"http://{host}:{port}"
        
        retry_config = RetryConfig(
            max_attempts=llm_config.get("retry_attempts", 3),
            base_delay=llm_config.get("retry_delay", 2.0),
        )
        
        return cls(
            base_url=base_url,
            endpoint=llm_config.get("endpoint", "/v1/chat/completions"),
            model=llm_config.get("model", "local-model"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048),
            top_p=llm_config.get("top_p", 0.9),
            timeout=llm_config.get("timeout", 120.0),
            retry_config=retry_config,
            app_config=app_config,
        )
    
    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry - create configured HTTP client."""
        # For local LLM, we typically don't need SSL or proxy
        # But we still use the configured client for consistency
        self._client = create_httpx_client(
            config=self.app_config,
            timeout=self.timeout,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Generate completion from prompt.
        
        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            **kwargs: Override default parameters
            
        Returns:
            Generated text completion
            
        Raises:
            LLMError: If generation fails after retries
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return await self.chat_completion(messages, **kwargs)
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> str:
        """
        Send chat completion request with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override default parameters
            
        Returns:
            Assistant's response text
        """
        # Ensure client exists
        if not self._client:
            self._client = create_httpx_client(
                config=self.app_config,
                timeout=self.timeout,
            )
        
        # Build request payload
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": False,
        }
        
        # Attempt with retries
        last_error: Exception | None = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                response = await self._client.post(
                    f"{self.base_url}{self.endpoint}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract response text
                choices = data.get("choices", [])
                if not choices:
                    raise LLMError("No choices in LLM response")
                
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                if not content:
                    raise LLMError("Empty content in LLM response")
                
                usage = data.get("usage", {})
                logger.info(
                    f"✅ LLM generation successful "
                    f"(prompt: {usage.get('prompt_tokens', 'N/A')}, "
                    f"completion: {usage.get('completion_tokens', 'N/A')} tokens)"
                )
                
                return content.strip()
                
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                
                if status == 503:  # Server overloaded
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"🔄 LLM server overloaded (503), "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_config.max_attempts})"
                    )
                    await asyncio.sleep(delay)
                elif status >= 500:  # Other server errors
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"🔄 LLM server error ({status}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(f"LLM HTTP error {status}: {e}")
                    
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"🔄 LLM connection error, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.retry_config.max_attempts}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"❌ LLM server not reachable at {self.base_url}\n"
                        f"   Troubleshooting:\n"
                        f"   1. Ensure llama.cpp server is running\n"
                        f"   2. Check if LLM_HOST and LLM_PORT environment variables are correct\n"
                        f"   3. Verify firewall allows connection to {self.base_url}\n"
                        f"   4. Try: curl {self.base_url}/health"
                    )
                    raise LLMConnectionError(
                        f"LLM server not reachable at {self.base_url}: {e}"
                    )
                    
            except httpx.ReadTimeout as e:
                last_error = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"⏱️  LLM request timed out after {self.timeout}s, "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                last_error = e
                logger.warning(
                    f"⚠️  LLM error on attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                if attempt < self.retry_config.max_attempts - 1:
                    await asyncio.sleep(self.retry_config.get_delay(attempt))
        
        raise LLMError(
            f"LLM request failed after {self.retry_config.max_attempts} attempts: {last_error}"
        )
    
    async def health_check(self) -> bool:
        """
        Check if LLM server is available with detailed diagnostics.
        
        Returns:
            True if server is responding, False otherwise
        """
        client = self._client or create_httpx_client(
            config=self.app_config,
            timeout=5.0,
        )
        close_client = self._client is None
        
        try:
            # Try health endpoint first
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info(f"✅ LLM server health check passed: {self.base_url}")
                    return True
            except httpx.RequestError:
                pass
            
            # Try models endpoint (OpenAI-compatible)
            try:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    logger.info(f"✅ LLM server responding (models endpoint): {self.base_url}")
                    return True
            except httpx.RequestError:
                pass
            
            # Try root endpoint
            try:
                response = await client.get(self.base_url)
                if response.status_code in [200, 404]:  # 404 still means server is up
                    logger.info(f"✅ LLM server responding: {self.base_url}")
                    return True
            except httpx.RequestError:
                pass
            
            # All checks failed
            logger.warning(
                f"❌ LLM server health check failed: {self.base_url}\n"
                f"   Troubleshooting steps:\n"
                f"   1. Start llama.cpp server: ./server -m model.gguf -c 4096\n"
                f"   2. Check LLM_HOST (current: {self.app_config.llm.host})\n"
                f"   3. Check LLM_PORT (current: {self.app_config.llm.port})\n"
                f"   4. Test manually: curl {self.base_url}/health"
            )
            return False
                
        finally:
            if close_client and client:
                await client.aclose()
    
    async def get_model_info(self) -> dict[str, Any] | None:
        """
        Get information about available models.
        
        Returns:
            Model information dict or None if unavailable
        """
        client = self._client or create_httpx_client(
            config=self.app_config,
            timeout=10.0,
        )
        close_client = self._client is None
        
        try:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None
        finally:
            if close_client and client:
                await client.aclose()


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without actual LLM server.
    
    Generates simple placeholder narrations.
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        # Don't call parent __init__ to avoid config logging
        self.base_url = "http://localhost:8080"
        self.endpoint = "/v1/chat/completions"
        self.model = "mock-model"
        self.temperature = 0.7
        self.max_tokens = 2048
        self.top_p = 0.9
        self.timeout = 10.0
        self.retry_config = RetryConfig()
        self.app_config = get_config()
        self._client = None
        self._mock_enabled = True
        
        logger.info("🎭 Using Mock LLM Client (no actual LLM server required)")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> str:
        """Generate mock response based on prompt content."""
        logger.info("🎭 Mock LLM generating response")
        
        # Extract some context from the prompt for more realistic mock
        prompt_lower = prompt.lower()
        
        if "trip" in prompt_lower or "journey" in prompt_lower:
            return self._generate_trip_narration()
        else:
            return self._generate_generic_response()
    
    def _generate_trip_narration(self) -> str:
        """Generate a mock trip narration."""
        return (
            "The journey began on a clear morning as the driver departed from their "
            "starting location. The route took them through a mix of urban streets "
            "and suburban roads. Throughout the trip, the driver maintained a steady "
            "pace, with occasional stops at traffic signals.\n\n"
            "The driving style was generally cautious, with smooth acceleration and "
            "gradual braking patterns. There were a few instances of moderate speed "
            "variations as the road conditions changed. The driver navigated several "
            "turns and intersections with care.\n\n"
            "As the trip progressed, the traffic conditions varied from light to "
            "moderate. The driver adapted their speed accordingly, showing good "
            "awareness of the surrounding environment. The journey concluded safely "
            "at the destination after covering the planned route."
        )
    
    def _generate_generic_response(self) -> str:
        """Generate a generic mock response."""
        return (
            "The driver started the journey and traveled through the route. "
            "The trip included various speed changes and turns along the way. "
            "The driver maintained generally safe driving behavior throughout the journey."
        )
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> str:
        """Generate mock chat completion."""
        # Extract the last user message for context
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        return await self.generate(user_message)
    
    async def health_check(self) -> bool:
        """Always returns True for mock client."""
        return True
    
    async def get_model_info(self) -> dict[str, Any]:
        """Return mock model info."""
        return {
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "owned_by": "trip-verbalizer",
                }
            ]
        }


class FallbackNarrator:
    """
    Fallback narrator when LLM is unavailable.
    
    Generates basic narrations using templates without LLM.
    """
    
    def __init__(self):
        logger.warning(
            "⚠️  Using FallbackNarrator - no LLM server available\n"
            "   Narrations will be template-based without AI enhancement"
        )
    
    def narrate(
        self,
        distance_km: float = 0,
        duration_min: float = 0,
        start_location: str = "unknown",
        end_location: str = "unknown",
        avg_speed_kmh: float = 0,
        events: list[str] | None = None,
    ) -> str:
        """
        Generate a basic template-based narration.
        
        Args:
            distance_km: Trip distance in kilometers
            duration_min: Trip duration in minutes
            start_location: Starting location name
            end_location: Ending location name
            avg_speed_kmh: Average speed in km/h
            events: List of notable events
            
        Returns:
            Basic trip narration
        """
        # Build basic narration
        parts = []
        
        # Opening
        if start_location != "unknown":
            parts.append(f"The trip started from {start_location}")
        else:
            parts.append("The trip started")
        
        if end_location != "unknown" and end_location != start_location:
            parts.append(f"and ended at {end_location}")
        
        # Distance and duration
        if distance_km > 0 and duration_min > 0:
            parts.append(
                f"covering approximately {distance_km:.1f} km "
                f"over {duration_min:.0f} minutes"
            )
        elif distance_km > 0:
            parts.append(f"covering approximately {distance_km:.1f} km")
        elif duration_min > 0:
            parts.append(f"lasting approximately {duration_min:.0f} minutes")
        
        # Speed
        if avg_speed_kmh > 0:
            parts.append(f"at an average speed of {avg_speed_kmh:.0f} km/h")
        
        narration = ", ".join(parts) + ".\n\n"
        
        # Events
        if events:
            narration += "Notable events during the trip:\n"
            for event in events[:5]:  # Limit to 5 events
                narration += f"- {event}\n"
        
        return narration
