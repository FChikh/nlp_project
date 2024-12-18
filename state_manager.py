# src/state_manager.py

import json
import os
from typing import List, Dict, Any


class ConversationState:
    """
    Class to manage the state of a conversation, including user and bot history,
    detected intents and slots, and database results."""
    def __init__(self):
        self.user_history = []
        self.bot_history = []
        self.domain: str = None
        self.intents: Dict[str, List[str]] = {}
        self.slots: Dict[str, Dict[str, List[str]]] = {}
        self.db_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def update_user_domain(self, domain: str):
        """
        Update the domain of the conversation and reset intents, slots, and database results if the domain changes.
        
        Parameters:
        - domain (str): The domain of the conversation.
        """
        if self.domain is not None and self.domain != domain:
            self.intents = {}
            self.slots = {}
            self.db_results = {}
        self.domain = domain

    def update_user_input(self, user_query: str):
        """
        Update the user's input history.
        
        Parameters:
        - user_query (str): The user's input message.
        """
        self.user_history.append(user_query)

    def update_bot_response(self, bot_response: str):
        """
        Update the bot's response history.
        
        Parameters:
        - bot_response (str): The bot's response message.
        """
        self.bot_history.append(bot_response)

    def update_intents_slots(self, domain: str, intent: str, slots: dict):
        """
        Update the detected intents and slots for a given domain.
        
        Parameters:
        - domain (str): The domain of the detected intents and slots.
        - intent (str): The detected intent.
        - slots (dict): The detected slots and their values.
        """
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
        """
        Update the database results for a given domain.
        
        Parameters:
        - domain (str): The domain of the database results.
        - db_data (List[Dict[str, Any]]): The retrieved database entries.
        """
        self.db_results[domain] = db_data

    def to_dict(self):
        return {
            "user_history": self.user_history,
            "bot_history": self.bot_history,
            "intents": self.intents,
            "slots": self.slots,
            "db_results": self.db_results
        }
