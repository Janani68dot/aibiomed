"""
Authentication module for Neural Screening System.
Handles patient & doctor login/registration using a local JSON store.
"""
import json
import hashlib
import os
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        # Seed default doctor account
        default = {
            "users": [
                {
                    "username": "doctor",
                    "password_hash": _hash("doctor123"),
                    "role": "doctor",
                    "name": "Dr. Aisha Patel",
                    "age": None,
                    "gender": None,
                    "created_at": datetime.now().isoformat()
                }
            ]
        }
        save_users(default)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(data: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def authenticate(username: str, password: str, role: str):
    """
    Returns user dict if credentials match, else None.
    """
    data = load_users()
    ph = _hash(password)
    for u in data["users"]:
        if u["username"] == username and u["password_hash"] == ph and u["role"] == role:
            return u
    return None

def register_patient(name: str, age: int, gender: str, username: str, password: str):
    """
    Registers a new patient. Returns (True, message) or (False, error).
    """
    data = load_users()
    # Check duplicate
    for u in data["users"]:
        if u["username"] == username:
            return False, "Username already taken. Please choose another."
    new_user = {
        "username": username,
        "password_hash": _hash(password),
        "role": "patient",
        "name": name,
        "age": int(age),
        "gender": gender,
        "created_at": datetime.now().isoformat()
    }
    data["users"].append(new_user)
    save_users(data)
    return True, "Registration successful!"

def get_all_patients():
    """Returns list of all patient user dicts."""
    data = load_users()
    return [u for u in data["users"] if u["role"] == "patient"]
