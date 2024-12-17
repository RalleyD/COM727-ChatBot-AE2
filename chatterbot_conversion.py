import yaml
import json
import os

def convert_chatterbot_to_intents(yaml_path, json_path):
    intents = {"intents" : []}

    # iterate over all yaml files in the chatterbot dataset
    for root, _, files in os.walk(yaml_path):
        for file in files:
            if file.endswith(".yml"):
                with open(os.path.join(root, file), 'r') as f:
                    data = yaml.safe_load(f)
                    # Each YAML file has categories and conversations
                    category = data.get('categories', ['general'])[0]
                    conversations = data.get('conversations',[])

                    # Create an intent for the category
                    patterns = []
                    responses = []
                    for convo in conversations:
                        if len(convo) >= 2:     # Must have both input and response 
                            patterns.append(convo[0]) # User input
                            responses.append(convo[1]) # Bot repsoinse

                    if patterns and responses:
                        intents["intents"].append({
                            "tag": category,
                            "patterns": patterns,
                            "responses": responses
                        })

    # Save as JSON
    with open(json_path, 'w') as f:
        json.dump(intents, f, indent=4)
    print(f"Converted data saved to {json_path}")

convert_chatterbot_to_intents("chatterbot", "new_intents.json")