from __future__ import annotations

import os
import requests
from typing import Optional


class FredClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        # Try Streamlit secrets first (when running in Streamlit)
        if api_key is None:
            try:
                import streamlit as st
                api_key = st.secrets.get("FRED_API_KEY")
            except Exception:
                api_key = None

        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing FRED API key. Set it in .streamlit/secrets.toml as FRED_API_KEY "
                "or export FRED_API_KEY in your terminal."
            )
        self.timeout = timeout
        self.base = "https://api.stlouisfed.org/fred"

    def latest_observation(self, series_id: str) -> tuple[str, float]:
        """
        Returns (date_str YYYY-MM-DD, value_float) for the latest non-missing observation.
        """
        url = f"{self.base}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        }
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        for obs in data.get("observations", []):
            d = obs.get("date")
            v = obs.get("value")
            if not d or v in (None, "."):
                continue
            try:
                return d, float(v)
            except ValueError:
                continue

        raise RuntimeError(f"No valid observations found for series {series_id}")
