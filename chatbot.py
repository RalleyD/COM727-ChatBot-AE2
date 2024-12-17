import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

json_files= ['Anaerobic_respiration.json',
             'Aerobic_respiration.json',
             'Gas_exchage.json','Greetings.json',
             'Response_to_exercise.json',
             'Type_of_respiration.json']
intents=[]
for file in json_files:
    intents.append(json.loads(open(file).read()))


words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/chatbot_model.keras')

#Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts the sentences into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) #bow: Bag Of Words, feed the data into the neural network
    res = model.predict(np.array([bow]))[0] #res: result. [0] as index 0
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
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



def main():
    print("COM727 Chatbot is here!")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        ints = predict_class(message)
        print(f"DEBUG: predicted intents: {ints}") # debug line
        res = get_response(ints, intents)
        print("Chatbot: ", res)

# Only start the chatbot if the script is run directly
if __name__ == "__main__":
    main()
