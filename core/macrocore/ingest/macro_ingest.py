from __future__ import annotations

from pathlib import Path

from core.macrocore.sources.fred import FredClient
from core.macrocore.storage.db import connect, utc_now_iso


def upsert_macro_observation(
    db_path: Path,
    series_id: str,
    obs_date: str,
    value: float,
    source: str,
) -> None:
    fetched_at = utc_now_iso()
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO macro_observations(series_id, obs_date, value, source, fetched_at_utc)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series_id, obs_date)
            DO UPDATE SET value=excluded.value, source=excluded.source, fetched_at_utc=excluded.fetched_at_utc
            """,
            (series_id, obs_date, value, source, fetched_at),
        )


def ingest_latest_fred_series(db_path: Path, series_id: str) -> dict:
    fred = FredClient()  # uses st.secrets["FRED_API_KEY"]
    obs_date, value = fred.latest_observation(series_id)

    upsert_macro_observation(
        db_path=db_path,
        series_id=series_id,
        obs_date=obs_date,
        value=value,
        source="FRED",
    )

    return {"series_id": series_id, "obs_date": obs_date, "value": value}
