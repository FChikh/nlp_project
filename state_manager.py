# src/state_manager.py

import json
import os
from typing import List, Dict, Any


class ConversationState:
    def __init__(self):
        self.user_history = []
        self.bot_history = []
        self.domain: str = None
        self.intents: Dict[str, List[str]] = {}
        self.slots: Dict[str, Dict[str, List[str]]] = {}
        self.db_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def update_user_domain(self, domain: str):
        if self.domain is not None and self.domain != domain:
            self.intents = {}
            self.slots = {}
            self.db_results = {}
        self.domain = domain

    def update_user_input(self, user_query: str):
        self.user_history.append(user_query)

    def update_bot_response(self, bot_response: str):
        self.bot_history.append(bot_response)

    def update_intents_slots(self, domain: str, intent: str, slots: dict):
        if domain not in self.intents:
            self.intents[domain] = []
        self.intents[domain].append(intent)

        if domain not in self.slots:
            self.slots[domain] = {}
        for slot, values in slots.items():
            if slot not in self.slots[domain]:
                self.slots[domain][slot] = []
            self.slots[domain][slot].extend(values)
        
    def update_db_results(self, domain: str, db_data):
        self.db_results[domain] = db_data

    def to_dict(self):
        return {
            "user_history": self.user_history,
            "bot_history": self.bot_history,
            "intents": self.intents,
            "slots": self.slots,
            "db_results": self.db_results
        }

    def save_state(self, directory: str = "conversations/"):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{self.user_id}.json"), 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
