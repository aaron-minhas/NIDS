"""
Live / replay-mode flow simulator.

Replay mode: stream rows from a CSV at a configurable rate. Used by the
"Live Demo" tab in the dashboard so the viva panel sees real-time-style
detection without needing actual packet capture.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

from src.preprocessing import clean_columns


def replay_csv(csv_path, flows_per_second=25.0, chunk_size=1, shuffle=True, seed=42):
    df = pd.read_csv(csv_path, low_memory=False)
    df = clean_columns(df)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    interval = chunk_size / max(flows_per_second, 0.001)
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        yield chunk
        time.sleep(interval)


def replay_csv_iter(csv_path, flows_per_second=25.0, shuffle=True, seed=42):
    for chunk in replay_csv(csv_path, flows_per_second=flows_per_second, chunk_size=1, shuffle=shuffle, seed=seed):
        yield chunk.iloc[0]


def scapy_live_capture_stub():
    return (
        "Live Scapy capture requires:\n"
        "  1. pip install scapy\n"
        "  2. Run with admin/root privileges (Npcap on Windows, libpcap on Linux).\n"
        "  3. Subclass scapy.AsyncSniffer with a flow-key dict on the 5-tuple.\n"
        "  4. Aggregate per-flow stats until flow timeout (120s) or FIN/RST.\n"
        "  5. Emit each finalised flow as a CICIDS-shaped row to NIDSDetector.\n"
        "  Replay mode is sufficient for AIES viva and avoids privilege/dependency tax."
    )
