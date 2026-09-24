"""naviguard.errors — shared exception types (mapped to HTTP 503 by the API)."""


class TelemetryNotFoundError(RuntimeError):
    """Raised when the telemetry CSV hasn't been generated yet."""


class ArtifactsInvalidError(RuntimeError):
    """Raised when model artifacts exist but are inconsistent with each other."""
