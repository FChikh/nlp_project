from domain_classifier import DomainClassifier
from intents_slots_extractor import IntentsSlotsExtractor
from retrieval import retrieve_db_entries
from response_generator import generate_response_t5
from schema_parser import SchemaParser
from state_manager import ConversationState


def main():
    schema_path = "multiwoz/data/MultiWOZ_2.2/schema.json"
    embedding_dir = "models/embeddings"
    retrieval_dir = "models/retrieval"
    domain_classifier_path = "./domain_classifier"

    schema_parser = SchemaParser(schema_path)
    domain_classifier = DomainClassifier(model_path=domain_classifier_path)
    intents_slots_extractor = IntentsSlotsExtractor(
        schema_parser,
        embedding_model_name='all-MiniLM-L6-v2',
        embeddings_dir=embedding_dir,
        intent_threshold=0.3,
        categorical_slot_threshold=0.4,
        non_categorical_slot_threshold=0.4
    )

    print("Welcome to the Chatbot! Type 'exit' to quit.")
    state = ConversationState()
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        # Update state with user input
        state.update_user_input(user_query)

        # Step 1: Domain Classification
        predicted_domains = domain_classifier.predict(user_query)
        if not predicted_domains:
            response = "I'm sorry, I couldn't determine the domain of your request."
            print(f"Chatbot: {response}")
            state.update_bot_response(response)
            continue
        domain = predicted_domains[0]
        state.update_user_domain(domain)

        # Step 2: Intent and Slot Extraction
        services = [domain]  # Assuming single domain for simplicity
        extracted = intents_slots_extractor.extract_intents_slots(
            user_query, services)
        print(f"[DEBUG] Extracted Intents and Slots: {extracted}")

        if domain in extracted:
            intent = extracted[domain].get("intent", "none")
            slots = extracted[domain].get("slots", {})
            slots_dict = {}
            for slot in slots:
                if ':' in slot:
                    slot_name, slot_value = slot.split(':', 1)
                    slots_dict.setdefault(slot_name, []).append(slot_value)
        else:
            intent = "none"
            slots_dict = {}

        state.update_intents_slots(domain, intent, slots_dict)

        # Step 3: DB Retrieval
        if domain != "none" and ("find" in intent or "book" in intent and state.db_results == {}):
            retrieved = retrieve_db_entries(
                user_query, domain, top_k=1, save_dir=retrieval_dir)
            state.update_db_results(domain, retrieved)
            print(type(state.db_results))

        # Step 4: Generate Response
        conversation_history = []
        for user, bot in zip(state.user_history[:-1], state.bot_history):
            conversation_history.append({"user": user, "bot": bot})

        response = generate_response_t5(
            user_query=user_query,
            domain=state.domain,
            intents=state.intents,
            slots=state.slots,
            db_results=state.db_results,
            conversation_history=conversation_history
        )
        print(f"Chatbot: {response}\n")

        # Update state with bot response
        state.update_bot_response(response)


if __name__ == "__main__":
    main()
