"""naviguard.llm.client — thin client for a local LM Studio server.

LM Studio exposes an OpenAI-compatible REST API on localhost once the user
loads a model and starts its local server (default http://localhost:1234/v1).
This wraps the official `openai` SDK pointed at that base_url — the
documented, supported way to talk to LM Studio — and never crashes the
caller: any connectivity/timeout/API failure is normalized into
LMStudioUnavailableError so callers can degrade gracefully.
"""

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from naviguard.config import LLMSettings


class LMStudioUnavailableError(RuntimeError):
    """Raised when LM Studio's local server isn't reachable or errors out."""


class LMStudioClient:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
        self._client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.timeout_s,
        )

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.settings.model,
                temperature=0.2,
                max_tokens=self.settings.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, APIError) as e:
            raise LMStudioUnavailableError(
                f"LM Studio not reachable at {self.settings.base_url}. "
                "Start LM Studio, load a model, and start the local server (Developer tab -> Start Server)."
            ) from e
