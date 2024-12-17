import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
intents = [json.loads(open('new_intents.json').read())]

# Load the JSON file
json_files= ['Anaerobic_respiration.json','Aerobic_respiration.json','Gas_exchage.json','Greetings.json','Response_to_exercise.json','Type_of_respiration.json']
for file in json_files:
    intents.append(json.loads(open(file).read()))
 
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', '/', '@']

for json_object in intents:
    for intent in json_object['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

stop_words = set(stopwords.words('english'))
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

# Create the training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

# Split into input and output
X = np.array(list(training[:, 0]))
y = np.array(list(training[:, 1]))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))

# gradient_descent_v2.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
hist = model.fit(
    X_train, y_train,
    epochs=200, batch_size=5,
    verbose=1,
    validation_data=(X_test, y_test)  # Evaluate on test set during training
)
# Save the model
model.save('models/chatbot_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

