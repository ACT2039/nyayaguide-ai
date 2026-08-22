"""
NyayaGuide AI — OpenRouter LLM Client
Handles secure communication with the OpenRouter API for LLM inference.
"""
from typing import List, Dict, Optional, Any
import requests

from ..config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT,
    is_openrouter_configured,
    get_safe_openrouter_status
)


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.
    Handles authentication, payload generation, timeouts, and error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = OPENROUTER_TIMEOUT
    ):
        self.api_key = (api_key if api_key is not None else OPENROUTER_API_KEY).strip()
        self.model = (model if model is not None else OPENROUTER_MODEL).strip()
        self.base_url = (base_url or OPENROUTER_BASE_URL).strip()
        self.timeout = timeout

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Sends chat completion request to OpenRouter and returns the generated content.
        """
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured. Please set OPENROUTER_API_KEY in your .env file."
            )

        active_model = (model if model is not None else self.model).strip()
        if not active_model:
            raise ValueError(
                "OPENROUTER_MODEL is not configured. Please specify a model in .env or arguments."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/charantejarangi123/nyayaguide_ai",
            "X-Title": "NyayaGuide AI"
        }

        payload: Dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if extra_params:
            payload.update(extra_params)

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            # Handle HTTP errors cleanly without revealing API key in logs
            if response.status_code == 401:
                raise PermissionError("OpenRouter authentication failed (401 Unauthorized). Please verify your OPENROUTER_API_KEY.")
            elif response.status_code == 402:
                raise RuntimeError("OpenRouter payment/credit required (402 Payment Required). Please check your OpenRouter account balance.")
            elif response.status_code == 429:
                raise RuntimeError("OpenRouter rate limit reached (429 Too Many Requests). Please wait before retrying.")
            
            response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"OpenRouter returned empty choices: {data}")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            if not content:
                raise RuntimeError("OpenRouter returned empty content in response.")

            return content.strip()

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"OpenRouter API request timed out after {self.timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            # Mask any potential credential exposure in exception message
            sanitized_err = str(e)
            if self.api_key and self.api_key in sanitized_err:
                sanitized_err = sanitized_err.replace(self.api_key, "***")
            raise RuntimeError(f"OpenRouter API network/request error: {sanitized_err}") from e
