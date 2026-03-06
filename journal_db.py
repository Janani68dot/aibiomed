"""
journal_db.py — Daily Memory Journal storage module.
Stores daily speech recordings and cognitive analysis results.
New module — does NOT touch existing auth.py, history.py, or users.json.
"""
import json
import os
from datetime import datetime, timedelta

JOURNAL_DIR = os.path.join(os.path.dirname(__file__), "journal_data")


def _journal_path(username: str) -> str:
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    return os.path.join(JOURNAL_DIR, f"{username}_journal.json")


def _load_journal(username: str) -> list:
    path = _journal_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_journal(username: str, entries: list):
    path = _journal_path(username)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def save_journal_entry(username: str, entry: dict):
    """
    Save a daily memory journal entry.
    Entry fields:
        date, transcript, audio_path,
        hesitation_score, fluency_score, vocabulary_score,
        repetition_score, cognitive_score
    """
    entries = _load_journal(username)
    entry["timestamp"] = datetime.now().isoformat()
    entry["user_id"] = username
    entries.append(entry)
    _save_journal(username, entries)


def get_journal_entries(username: str, days: int = 7) -> list:
    """Return entries from the last `days` days."""
    entries = _load_journal(username)
    cutoff = datetime.now() - timedelta(days=days)
    recent = [
        e for e in entries
        if datetime.fromisoformat(e["timestamp"]) > cutoff
    ]
    return sorted(recent, key=lambda e: e["timestamp"])


def already_recorded_today(username: str) -> bool:
    """Check if a journal entry exists for today."""
    entries = _load_journal(username)
    today = datetime.now().date().isoformat()
    for e in entries:
        try:
            if e["timestamp"][:10] == today:
                return True
        except Exception:
            pass
    return False


def get_all_journal_users() -> list:
    """Return list of usernames that have any journal data."""
    if not os.path.exists(JOURNAL_DIR):
        return []
    users = []
    for fname in os.listdir(JOURNAL_DIR):
        if fname.endswith("_journal.json"):
            users.append(fname.replace("_journal.json", ""))
    return users


def export_journal_csv(username: str, days: int = 30) -> str:
    """Return a CSV string of journal entries for the last N days."""
    entries = get_journal_entries(username, days=days)
    if not entries:
        return "date,cognitive_score,hesitation_score,fluency_score,vocabulary_score,repetition_score,transcript\n"
    
    lines = ["date,cognitive_score,hesitation_score,fluency_score,vocabulary_score,repetition_score,transcript"]
    for e in entries:
        date = e.get("timestamp", "")[:10]
        cs = e.get("cognitive_score", 0)
        hs = e.get("hesitation_score", 0)
        fs = e.get("fluency_score", 0)
        vs = e.get("vocabulary_score", 0)
        rs = e.get("repetition_score", 0)
        tx = e.get("transcript", "").replace(",", " ").replace("\n", " ")
        lines.append(f"{date},{cs:.4f},{hs:.4f},{fs:.4f},{vs:.4f},{rs:.4f},{tx}")
    return "\n".join(lines)
