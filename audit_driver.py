"""Execution entrypoint for the standalone 300 x 24-hour benchmark."""
import argparse
import pandas as pd
import audit_300_24h as audit

_original_grid = audit.future_grid

def normalized_grid(start, end):
    # ISO timestamps restore fixed offsets; normalize both to the exchange timezone
    # before combining them with scheduled NY timestamps. Values remain unchanged.
    start = pd.Timestamp(start).tz_convert('America/New_York')
    end = pd.Timestamp(end).tz_convert('America/New_York')
    result = _original_grid(start, end)
    if not isinstance(result, pd.DatetimeIndex) or result.tz is None:
        raise ValueError('Future timestamps lost timezone-aware datetime dtype')
    if len(result) != 17 or not result.is_monotonic_increasing or not result.is_unique:
        raise ValueError('Malformed future exchange-session grid')
    if end-start != pd.Timedelta(hours=24):
        raise ValueError('Forecast must cover exactly 24 elapsed hours')
    return result

audit.future_grid = normalized_grid
for ds in ['2025-03-06', '2025-04-01', '2026-09-17']:
    start=pd.Timestamp(ds+' 16:00',tz='America/New_York')
    end=start+pd.Timedelta(hours=24)
    grid=normalized_grid(str(start),str(end))
    assert grid[-1] == end-pd.Timedelta(minutes=30)
    assert grid[0] == start
    assert pd.Series(grid).dt.hour.notna().all()

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',default='evidence/TSLA_1h_extended_raw.csv')
    ap.add_argument('--model',choices=['small','mini'],required=True)
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--out',default='results')
    audit.run(ap.parse_args())
