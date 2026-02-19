"""
LLM Client Module

This module handles communication with the local llama.cpp server
for generating natural language narrations.
"""

import asyncio
import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised when LLM request fails."""
    pass


class LLMClient:
    """
    Async client for llama.cpp HTTP API.
    
    Sends prompts to local LLM server and retrieves completions.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        endpoint: str = "/v1/chat/completions",
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        timeout: float = 120.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: llama.cpp server base URL
            endpoint: Completion endpoint path
            model: Model name (for API compatibility)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._client: httpx.AsyncClient | None = None
    
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
        
        return cls(
            base_url=llm_config.get("base_url", "http://localhost:8080"),
            endpoint=llm_config.get("endpoint", "/v1/chat/completions"),
            model=llm_config.get("model", "local-model"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048),
            top_p=llm_config.get("top_p", 0.9),
            timeout=llm_config.get("timeout", 120.0),
            retry_attempts=llm_config.get("retry_attempts", 3),
            retry_delay=llm_config.get("retry_delay", 2.0),
        )
    
    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
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
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override default parameters
            
        Returns:
            Assistant's response text
        """
        # Ensure client exists
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        close_client = self._client is None
        
        try:
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
            
            for attempt in range(self.retry_attempts):
                try:
                    response = await client.post(
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
                    
                    logger.info(
                        f"LLM generation successful "
                        f"(tokens: {data.get('usage', {}).get('completion_tokens', 'N/A')})"
                    )
                    
                    return content.strip()
                    
                except httpx.HTTPStatusError as e:
                    last_error = e
                    logger.warning(
                        f"LLM HTTP error on attempt {attempt + 1}: {e.response.status_code}"
                    )
                    
                except httpx.RequestError as e:
                    last_error = e
                    logger.warning(
                        f"LLM request error on attempt {attempt + 1}: {e}"
                    )
                
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"LLM error on attempt {attempt + 1}: {e}"
                    )
                
                # Wait before retry
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
            
            raise LLMError(
                f"LLM request failed after {self.retry_attempts} attempts: {last_error}"
            )
            
        finally:
            if close_client:
                await client.aclose()
    
    async def health_check(self) -> bool:
        """
        Check if LLM server is available.
        
        Returns:
            True if server is responding, False otherwise
        """
        client = self._client or httpx.AsyncClient(timeout=5.0)
        close_client = self._client is None
        
        try:
            # Try health endpoint first
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            
            # Try models endpoint (OpenAI-compatible)
            try:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            
            # Try root endpoint
            try:
                response = await client.get(self.base_url)
                return response.status_code in [200, 404]  # 404 still means server is up
            except httpx.RequestError:
                return False
                
        finally:
            if close_client:
                await client.aclose()
    
    async def get_model_info(self) -> dict[str, Any] | None:
        """
        Get information about available models.
        
        Returns:
            Model information dict or None if unavailable
        """
        client = self._client or httpx.AsyncClient(timeout=10.0)
        close_client = self._client is None
        
        try:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None
        finally:
            if close_client:
                await client.aclose()


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without actual LLM server.
    
    Generates simple placeholder narrations.
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._mock_enabled = True
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> str:
        """Generate mock response."""
        logger.info("Using mock LLM client")
        
        # Simple mock response
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
