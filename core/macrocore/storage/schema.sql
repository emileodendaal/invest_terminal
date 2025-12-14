PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS macro_observations (
  series_id TEXT NOT NULL,
  obs_date  TEXT NOT NULL,   -- YYYY-MM-DD
  value     REAL NOT NULL,
  source    TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,        -- CPI, NFP, FOMC, GDP...
  event_ts_utc TEXT NOT NULL,      -- ISO timestamp in UTC
  actual REAL,
  forecast REAL,
  previous REAL,
  units TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS prices (
  symbol TEXT NOT NULL,
  ts_utc TEXT NOT NULL,            -- ISO timestamp in UTC
  price REAL NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (symbol, ts_utc)
);

CREATE TABLE IF NOT EXISTS event_reactions (
  reaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  window TEXT NOT NULL,            -- "30m", "2h", "1d", ...
  ret REAL NOT NULL,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_observations(series_id, obs_date);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, event_ts_utc);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_time ON prices(symbol, ts_utc);
CREATE INDEX IF NOT EXISTS idx_reactions_event_symbol ON event_reactions(event_id, symbol);
