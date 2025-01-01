import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model


json_files = ['Anaerobic_respiration.json',
              'Aerobic_respiration.json',
              'Gas_exchage.json',
              'Greetings.json',
              'Response_to_exercise.json',
              'Type_of_respiration.json',
              ]
intents = []
for file in json_files:
    intents.append(json.loads(open(file).read()))

classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/chatbot_model2')


def predict_class(sentence):
    res = model.predict([sentence]).ravel().tolist()
    ERROR_THRESHOLD = 0.10
    print(f"result is {sorted(res)}")
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print(f"DEBUG: predicted intents: {return_list}")  # debug line
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        tag = 'default'
    else:
        tag = intents_list[0]['intent']
    for intent_json in intents_json:
        list_of_intents = intent_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
    return result


def handle_query(message):
    ints = predict_class(message)
    return get_response(ints, intents)


def main():
    print("COM727 Chatbot is here!")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        res = handle_query(message)
        print("Chatbot: ", res)


# Only start the chatbot if the script is run directly
if __name__ == "__main__":
    main()
