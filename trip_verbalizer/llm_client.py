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
        self.long_narration = False
        
        logger.info("🎭 Using Mock LLM Client (no actual LLM server required)")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> str:
        """Generate mock response based on prompt content."""
        logger.info("🎭 Mock LLM generating response")
        
        # Extract trip details from the prompt
        return self._generate_trip_narration(prompt)
    
    def _extract_field(self, prompt: str, field: str) -> str:
        """Extract a field value from the prompt (supports JSON format)."""
        import re
        import json as json_module
        
        # Try to parse as JSON first
        try:
            if "Input:" in prompt:
                json_str = prompt.split("Input:", 1)[1].strip()
                data = json_module.loads(json_str)
                
                # Navigate nested structure
                if field == "start":
                    return data.get("route", {}).get("start", "")
                elif field == "end":
                    return data.get("route", {}).get("end", "")
                elif field == "direction":
                    return data.get("route", {}).get("direction", "")
                elif field == "duration":
                    return data.get("trip", {}).get("duration", "")
                elif field == "distance":
                    return data.get("trip", {}).get("distance", "")
                elif field == "average_speed":
                    return data.get("speed", {}).get("average", "")
                elif field == "maximum_speed":
                    return data.get("speed", {}).get("maximum", "")
                elif field == "start_time":
                    return data.get("trip", {}).get("start_time", "")
                elif field == "major_roads":
                    return data.get("route", {}).get("major_roads", [])
                elif field == "events":
                    return data.get("events", [])
                elif field == "phases":
                    return data.get("phases", [])
        except (json_module.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Fallback to regex patterns
        patterns = [
            rf'^{field}:\s*([^\n]+)',
            rf'\n{field}:\s*([^\n]+)',
            rf'"{field}":\s*"([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_json_data(self, prompt: str) -> dict:
        """Extract full JSON data from prompt."""
        import json as json_module
        try:
            if "Input:" in prompt:
                json_str = prompt.split("Input:", 1)[1].strip()
                return json_module.loads(json_str)
        except (json_module.JSONDecodeError, KeyError, TypeError):
            pass
        return {}
    
    def _extract_events(self, prompt: str) -> list[str]:
        """Extract events from the prompt."""
        import re
        events = []
        # Look for event patterns
        event_pattern = r'Event[:\s]+([^\n]+)|event_type[:\s]+([^\n]+)'
        for match in re.finditer(event_pattern, prompt, re.IGNORECASE):
            event = match.group(1) or match.group(2)
            if event:
                events.append(event.strip())
        return events
    
    def _generate_trip_narration(self, prompt: str) -> str:
        """Generate a contextual trip narration using prompt data."""
        # Try to extract from JSON format first
        data = self._extract_json_data(prompt)
        
        if data:
            return self._generate_from_json(data)
        
        # Fallback to field extraction
        start_location = self._extract_field(prompt, "start")
        end_location = self._extract_field(prompt, "end")
        duration = self._extract_field(prompt, "duration")
        distance = self._extract_field(prompt, "distance")
        avg_speed = self._extract_field(prompt, "average_speed")
        max_speed = self._extract_field(prompt, "maximum_speed")
        start_time = self._extract_field(prompt, "start_time")
        direction = self._extract_field(prompt, "direction")
        
        return self._build_narration(
            start_location, end_location, duration, distance,
            avg_speed, max_speed, start_time, direction, [], []
        )
    
    def _generate_from_json(self, data: dict) -> str:
        """Generate narration from parsed JSON data."""
        trip = data.get("trip", {})
        route = data.get("route", {})
        speed = data.get("speed", {})
        events = data.get("events", [])
        phases = data.get("phases", [])
        driver = data.get("driver_behavior", {})
        
        return self._build_narration(
            start_location=route.get("start", ""),
            end_location=route.get("end", ""),
            duration=trip.get("duration", ""),
            distance=trip.get("distance", ""),
            avg_speed=speed.get("average", ""),
            max_speed=speed.get("maximum", ""),
            start_time=trip.get("start_time", ""),
            direction=route.get("direction", ""),
            events=events,
            phases=phases,
            major_roads=route.get("major_roads", []),
            areas=route.get("areas", []),
            driver_rating=driver.get("rating", ""),
            driver_style=driver.get("style", ""),
        )
    
    def _build_narration(
        self,
        start_location: str,
        end_location: str,
        duration: str,
        distance: str,
        avg_speed: str,
        max_speed: str,
        start_time: str,
        direction: str,
        events: list,
        phases: list,
        major_roads: list | None = None,
        areas: list | None = None,
        driver_rating: str = "",
        driver_style: str = "",
    ) -> str:
        """Build the narration from extracted data."""
        parts = []
        
        # Opening with start location and time
        if start_location and start_location.lower() != "unknown":
            parts.append(f"The journey began from {start_location}")
            if start_time:
                parts.append(f" at {start_time}")
        else:
            parts.append("The journey began")
        parts.append(".\n\n")
        
        # Movement and direction
        if direction:
            parts.append(f"The driver headed {direction.lower()}")
        else:
            parts.append("The driver proceeded along the route")
        
        # Mention roads if available
        if major_roads and len(major_roads) > 0:
            roads = [r for r in major_roads if r]
            if roads:
                if len(roads) == 1:
                    parts.append(f" via {roads[0]}")
                else:
                    parts.append(f" via {roads[0]} and {roads[1]}")
        
        if distance:
            parts.append(f", covering {distance}")
        
        if duration:
            parts.append(f" over {duration}")
        
        parts.append(". ")
        
        # Speed information
        if avg_speed:
            parts.append(f"The average speed was {avg_speed}")
            if max_speed:
                parts.append(f", reaching {max_speed} at peak")
            parts.append(".")
        
        # Areas passed (for long narration)
        if getattr(self, 'long_narration', False) and areas:
            area_names = [a for a in areas if a]
            if area_names:
                parts.append(f" The route passed through {', '.join(area_names[:3])}.")
        
        parts.append("\n\n")
        
        # Phases (for long narration)
        if getattr(self, 'long_narration', False) and phases:
            parts.append("The trip progressed through several phases: ")
            phase_descs = []
            for p in phases[:5]:
                if isinstance(p, dict):
                    ptype = p.get("type", "")
                    ptime = p.get("time", "")
                    ploc = p.get("location", "")
                    pdesc = p.get("description", "")
                    if ptype:
                        if ploc and ptime:
                            phase_descs.append(f"{ptype} at {ploc} ({ptime})")
                        elif pdesc:
                            phase_descs.append(pdesc)
                        else:
                            phase_descs.append(ptype)
            if phase_descs:
                parts.append(", ".join(phase_descs))
                parts.append(".\n\n")
        
        # Events
        if events:
            event_descriptions = []
            max_events = 5 if getattr(self, 'long_narration', False) else 3
            for e in events[:max_events]:
                if isinstance(e, dict):
                    desc = e.get("description", e.get("type", ""))
                    time = e.get("time", "")
                    severity = e.get("severity", "")
                    if desc:
                        event_str = desc
                        if time:
                            event_str = f"{desc} at {time}"
                        if getattr(self, 'long_narration', False) and severity:
                            event_str += f" ({severity})"
                        event_descriptions.append(event_str)
            
            if event_descriptions:
                if getattr(self, 'long_narration', False):
                    parts.append("Notable events during the trip included: ")
                    parts.append("; ".join(event_descriptions))
                    parts.append(".")
                else:
                    parts.append(f"During the trip, {event_descriptions[0].lower()}")
                    if len(event_descriptions) > 1:
                        parts.append(f", and {event_descriptions[1].lower()}")
                    parts.append(".")
                parts.append(" ")
        
        # Driving behavior
        if getattr(self, 'long_narration', False):
            if driver_style == "smooth":
                parts.append(
                    "The driving style was notably smooth and controlled, with gradual "
                    "acceleration and gentle braking patterns throughout the journey. "
                )
            elif driver_style == "aggressive":
                parts.append(
                    "The driving patterns showed some aggressive tendencies, with "
                    "rapid acceleration and hard braking at various points. "
                )
            else:
                parts.append(
                    "The driver maintained a balanced driving style, adapting to "
                    "traffic conditions and road changes appropriately. "
                )
            if driver_rating:
                parts.append(f"Overall driver rating: {driver_rating}. ")
        else:
            parts.append(
                "The driver maintained steady control throughout the journey"
            )
            parts.append(".")
        
        parts.append("\n\n")
        
        # Ending with destination
        if end_location and end_location.lower() != "unknown":
            parts.append(f"The trip concluded at {end_location}")
        else:
            parts.append("The trip concluded at the destination")
        
        # Add route summary if both locations are known and different
        if (end_location and start_location and 
            end_location.lower() != "unknown" and 
            start_location.lower() != "unknown"):
            start_short = start_location.split(',')[0].strip()
            end_short = end_location.split(',')[0].strip()
            if start_short != end_short:
                parts.append(f", completing the route from {start_short} to {end_short}")
        
        parts.append(".")
        
        # Summary for long narration
        if getattr(self, 'long_narration', False):
            parts.append(f"\n\nIn summary, this was a {duration or 'brief'} journey")
            if distance:
                parts.append(f" covering {distance}")
            parts.append(", with the driver demonstrating ")
            if driver_style == "smooth":
                parts.append("safe and efficient driving habits.")
            elif driver_style == "aggressive":
                parts.append("room for improvement in driving smoothness.")
            else:
                parts.append("generally responsible driving behavior.")
        
        return "".join(parts)
    
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
