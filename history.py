"""
History module for Neural Screening System.
Handles persistent storage and retrieval of cognitive assessment results.
"""
import json
import os
from datetime import datetime, timedelta

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "patient_history")

def save_assessment(username: str, report_data: dict):
    """
    Saves a single assessment for a user.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    filename = f"{username}_history.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    
    history = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []
    
    # Add metadata
    report_data['timestamp'] = datetime.now().isoformat()
    history.append(report_data)
    
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

def get_patient_history(username: str, days: int = 7):
    """
    Retrieves assessment history for a specific patient for the last N days.
    """
    filename = f"{username}_history.json"
    filepath = os.path.join(HISTORY_DIR, filename)
    
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, "r") as f:
        try:
            history = json.load(f)
        except:
            return []
            
    # Filter by date
    cutoff = datetime.now() - timedelta(days=days)
    recent_history = [
        entry for entry in history 
        if datetime.fromisoformat(entry['timestamp']) > cutoff
    ]
    
    return recent_history

def get_all_recent_assessments(days: int = 7):
    """
    Returns a mapping of username -> recent assessments for all patients.
    """
    if not os.path.exists(HISTORY_DIR):
        return {}
        
    all_data = {}
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith("_history.json"):
            username = filename.replace("_history.json", "")
            all_data[username] = get_patient_history(username, days)
    
    return all_data
