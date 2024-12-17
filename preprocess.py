# src/preprocess.py
import json
import os
import pandas as pd

# Paths (Adjust these if necessary)
BASE_PATH = "multiwoz/data/MultiWOZ_2.2"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")


def extract_dialogues(data_path):
    dialogues = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.json'):
            with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                dialogues.extend(data)
    return dialogues


def preprocess_dialogues(dialogues):
    records = []
    for d in dialogues:
        turns = d.get("turns", [])
        # We'll pair user utterance (USER) with the next system utterance (SYSTEM)
        for i, turn in enumerate(turns):
            if turn["speaker"] == "USER":
                user_utt = turn["utterance"]
                turn_id = int(turn["turn_id"])
                system_utt = ""
                if turn_id + 1 < len(turns) and turns[turn_id + 1]["speaker"] == "SYSTEM":
                    system_utt = turns[turn_id + 1]["utterance"]

                # Extract domains, intents, slot_values from frames
                active_domains = set()
                active_intents = set()
                slot_values = {}
                for frame in turn["frames"]:
                    service = frame["service"]
                    state = frame.get("state", {})
                    intent = state.get("active_intent", "NONE")
                    sv = state.get("slot_values", {})

                    if intent != "NONE":
                        active_intents.add(intent)
                        active_domains.add(service)
                        for k, v in sv.items():
                            slot_values[k] = v

                records.append({
                    "dialogue_id": d["dialogue_id"],
                    "user_utterance": user_utt,
                    "system_utterance": system_utt,
                    "domains": str(list(active_domains)),
                    "intents": str(list(active_intents)),
                    "slot_values": str(slot_values),
                })
    return pd.DataFrame(records)


def main():
    # Extract dialogues
    train_dialogues = extract_dialogues(TRAIN_PATH)
    test_dialogues = extract_dialogues(TEST_PATH)

    # Preprocess
    train_df = preprocess_dialogues(train_dialogues)
    test_df = preprocess_dialogues(test_dialogues)

    train_df.to_csv("train_preprocessed.csv", index=False)
    test_df.to_csv("test_preprocessed.csv", index=False)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
