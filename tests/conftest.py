"""Mock mode is the default for the test process. No Tinker key and no database."""
from __future__ import annotations

import os

os.environ["FIELDSIGHT_MOCK"] = "1"
os.environ.pop("DATABASE_URL", None)
os.environ.pop("TINKER_API_KEY", None)
