"""naviguard — AI-driven NavIC/GNSS satellite clock bias & ephemeris error intelligence."""

import sys as _sys

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode the unicode arrows/checkmarks used in status messages throughout
# this package; force UTF-8 on import so it works regardless of entry point
# (CLI, direct script execution, or the uvicorn-run API process).
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
