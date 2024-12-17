# src/schema_parser.py

import json
import os


class SchemaParser:
    def __init__(self, schema_path: str):
        """
        Initialize the SchemaParser with the path to the schema file.

        Parameters:
        - schema_path (str): Path to the schema JSON file.
        """
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        self.services = {service['service_name']
            : service for service in self.schema}

    def get_services(self):
        """
        Get a list of all service (domain) names.

        Returns:
        - List[str]: List of service names.
        """
        return list(self.services.keys())

    def get_intents(self, service_name):
        """
        Get intents for a given service.

        Parameters:
        - service_name (str): Name of the service/domain.

        Returns:
        - List[Dict[str, Any]]: List of intents with their details.
        """
        return self.services[service_name]['intents']

    def get_slots(self, service_name):
        """
        Get slots for a given service.

        Parameters:
        - service_name (str): Name of the service/domain.

        Returns:
        - List[Dict[str, Any]]: List of slots with their details.
        """
        return self.services[service_name]['slots']

    def get_slot_possible_values(self, service_name, slot_name):
        """
        Get possible values for a specific slot in a service.

        Parameters:
        - service_name (str): Name of the service/domain.
        - slot_name (str): Name of the slot.

        Returns:
        - List[str]: List of possible values. Empty if not categorical or no predefined values.
        """
        slots = self.get_slots(service_name)
        for slot in slots:
            if slot['name'] == slot_name:
                return slot.get('possible_values', [])
        return []

    def is_slot_categorical(self, service_name, slot_name):

        slots = self.get_slots(service_name)
        for slot in slots:
            if slot['name'] == slot_name:
                return slot.get('is_categorical', False)
        return False



if __name__ == "__main__":
    schema_path = "multiwoz/data/MultiWOZ_2.2/schema.json"
    schema_parser = SchemaParser(schema_path)

    services = schema_parser.get_services()
    for service in services:
        intents = schema_parser.get_intents(service)
        slots = schema_parser.get_slots(service)
        print(f"Service: {service}")
        print(f"Intents: {intents}")
        print(f"Slots: {slots}")