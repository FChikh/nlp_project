"""A simple chatbot interface using Streamlit to interact with the trained models."""

import streamlit as st
from domain_classifier import DomainClassifier
from intents_slots_extractor import IntentsSlotsExtractor
from retrieval import retrieve_db_entries
from response_generator import ResponseGenerator
from schema_parser import SchemaParser
from state_manager import ConversationState
import os


@st.cache_resource
def initialize_components():
    # Paths to necessary resources
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

    return domain_classifier, intents_slots_extractor, retrieval_dir

# Initialize components
domain_classifier, intents_slots_extractor, retrieval_dir = initialize_components()


@st.cache_resource
def initialize_response_generator():
    model_path = "./t5_finetuned"
    if not os.path.exists(model_path):
        st.warning(f"Model path {
                   model_path} does not exist. Please train the generator first.")
        return None
    return ResponseGenerator(model_path=model_path)


response_generator = initialize_response_generator()

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = ConversationState()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("🏠 Chatbot Interface")

st.markdown("""
**Instructions:**
- Type your message in the input box below and press "Send" to interact with the chatbot.
- Click "Clear Conversation" to reset the chat history.
""")

# Display conversation history
def display_conversation():
    for exchange in st.session_state.conversation_history:
        user_msg = exchange['user']
        bot_msg = exchange['bot']
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Chatbot:** {bot_msg}\n")


display_conversation()

# User input
user_input = st.text_input("You:", "", key="input")

if st.button("Send") and user_input.strip() != "":
    if response_generator is None:
        st.error(
            "Response Generator model not found. Please train the generator before interacting.")
    else:
        with st.spinner("Processing..."):
            st.session_state.conversation_state.update_user_input(user_input)

            predicted_domains = domain_classifier.predict(user_input)
            if not predicted_domains:
                response = "I'm sorry, I couldn't determine the domain of your request."
                st.session_state.conversation_history.append(
                    {"user": user_input, "bot": response})
                st.session_state.conversation_state.update_bot_response(
                    response)
                st.rerun()
            else:
                domain = predicted_domains[0]
                st.session_state.conversation_state.update_user_domain(domain)

                # Step 2: Intent and Slot Extraction
                services = [domain] # Assuming single domain for simplicity
                extracted = intents_slots_extractor.extract_intents_slots(
                    user_input, services)
                st.write(f"[DEBUG] Extracted Intents and Slots: {extracted}")

                if domain in extracted:
                    intent = extracted[domain].get("intent", "none")
                    slots = extracted[domain].get("slots", {})
                    slots_dict = {}
                    for slot in slots:
                        if ':' in slot:
                            slot_name, slot_value = slot.split(':', 1)
                            slots_dict.setdefault(
                                slot_name, []).append(slot_value)
                        else:
                            slots_dict.setdefault(slot, []).append("unknown")
                else:
                    intent = "none"
                    slots_dict = {}

                st.session_state.conversation_state.update_intents_slots(
                    domain, intent, slots_dict)

                # Step 3: DB Retrieval
                if domain != "none" and ("find" in intent or "book" in intent and st.session_state.conversation_state.db_results == {}):
                    retrieved = retrieve_db_entries(
                        user_input, domain, top_k=1, save_dir=retrieval_dir)
                    st.session_state.conversation_state.update_db_results(
                        domain, retrieved)
                    print(type(st.session_state.conversation_state.db_results))

                # Step 4: Generate Response
                conversation_history = st.session_state.conversation_history.copy()
                if conversation_history:
                    conversation_history = conversation_history[:-1]

                # Use the ResponseGenerator instance to generate a response
                response = response_generator.generate_response(
                    user_query=user_input,
                    domain=st.session_state.conversation_state.domain,
                    intents=st.session_state.conversation_state.intents.get(
                        domain, []),
                    slots=st.session_state.conversation_state.slots.get(
                        domain, {}),
                    db_results=st.session_state.conversation_state.db_results,
                    conversation_history=conversation_history
                )

                # Update conversation history
                st.session_state.conversation_history.append(
                    {"user": user_input, "bot": response})

                # Update state with bot response
                st.session_state.conversation_state.update_bot_response(
                    response)

                # Refresh the page to display the new message
                st.rerun()

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.session_state.conversation_state = ConversationState()
    st.rerun()
