"""The only module that talks to Gemini (google-genai SDK).

The API key is read from wearable-ml/.env (GEMINI_API_KEY) and passed straight
to the SDK. It is never printed, never logged and never put in an exception
message: every error text is passed through `redact` before it is kept.

SDK errors are mapped to three kinds the runner understands:
  RateLimited    429 per-minute limit: back off and retry
  QuotaExhausted 429 per-day / quota exhausted: stop the run and report
  StopRun        401 / 403 (key or permission): stop the run and report
  CallFailed     anything else (retryable when `retryable` is True: 5xx,
                 408, timeouts and dropped connections)
"""

import re
import time

from . import config


class CallFailed(Exception):
    def __init__(self, message, retryable=False, retry_after=None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class RateLimited(CallFailed):
    def __init__(self, message, retry_after=None):
        super().__init__(message, retryable=True, retry_after=retry_after)


class StopRun(CallFailed):
    """An error that makes every further call pointless: stop the whole run."""


class QuotaExhausted(StopRun):
    pass


def redact(text, secret):
    text = str(text)
    if secret:
        text = text.replace(secret, "[redacted]")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[redacted]", text)


def classify(exc, secret=None):
    """Map an SDK exception (anything with .code / message) to our error kinds."""
    message = redact(exc, secret)
    code = getattr(exc, "code", None)
    delay = re.search(r"retry(?:Delay|_delay)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", message)
    retry_after = float(delay.group(1)) if delay else None
    if code == 429:
        if re.search(r"per ?day|PerDay|daily", message, re.I):
            return QuotaExhausted(message)
        return RateLimited(message, retry_after=retry_after)
    if code in (401, 403):
        return StopRun(message)
    if code is not None and (int(code) == 408 or 500 <= int(code) < 600):
        return CallFailed(message, retryable=True, retry_after=retry_after)
    if code is None and re.search(r"timeout|timed out|connect|remoteprotocol|readerror",
                                  type(exc).__name__ + " " + message, re.I):
        return CallFailed(message, retryable=True)
    return CallFailed(message, retryable=False)


def load_api_key(env_file=config.ENV_FILE):
    from dotenv import dotenv_values
    key = dotenv_values(env_file).get(config.API_KEY_NAME)
    if not key:
        raise SystemExit("{} is not set in {} (the value is never printed).".format(
            config.API_KEY_NAME, env_file))
    return key


def sdk_version():
    try:
        import google.genai
        return "google-genai " + google.genai.__version__
    except ImportError:
        return "google-genai not installed"


class GeminiClient:
    """generate(system, user, settings) -> dict, timed, with token counts."""

    def __init__(self, model, api_key):
        from google import genai
        from google.genai import types
        self._key = api_key
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=config.SDK_RETRY_ATTEMPTS),
            timeout=config.REQUEST_TIMEOUT_S * 1000))      # milliseconds
        self.model = model
        self.sdk = sdk_version()

    def check_model(self):
        """Fail early if the model is not in the account's model list."""
        try:
            info = self._client.models.get(model=self.model)
        except Exception as exc:
            raise classify(exc, self._key)
        actions = getattr(info, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            raise CallFailed("{} does not support generateContent".format(self.model))
        return info.name

    def generate(self, system, user, settings):
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=settings["temperature"],
            max_output_tokens=settings["max_output_tokens"])
        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=cfg)
        except Exception as exc:
            raise classify(exc, self._key)
        latency = time.perf_counter() - start
        usage = response.usage_metadata
        candidate = (response.candidates or [None])[0]
        finish = getattr(candidate, "finish_reason", None)
        try:
            text = response.text
        except Exception:      # e.g. blocked: no text part
            text = None
        return {
            "text": text,
            "finish_reason": getattr(finish, "name", None) or (str(finish) if finish else None),
            "latency_s": latency,
            "tokens": {
                "prompt": getattr(usage, "prompt_token_count", None),
                "output": getattr(usage, "candidates_token_count", None),
                "thoughts": getattr(usage, "thoughts_token_count", None),
                "total": getattr(usage, "total_token_count", None),
            },
            "response_json": redact(response.model_dump_json(exclude_none=True), self._key),
        }
