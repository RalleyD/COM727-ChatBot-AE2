import json
import numpy as np
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TextVectorization, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
intents = [json.loads(open('new_intents.json').read())]

# Load the JSON files
json_files = ['Anaerobic_respiration.json', 'Aerobic_respiration.json',
              'Gas_exchage.json', 'Greetings.json',
              'Response_to_exercise.json', 'Type_of_respiration.json']
for file in json_files:
    intents.append(json.loads(open(file).read()))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', '/', '@']

# Process the intents
for json_object in intents:
    for intent in json_object['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

stop_words = set(stopwords.words('english'))
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
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
    X, y, test_size=0.2, random_state=42)

# Extract patterns and labels
all_patterns = []
all_labels = []

for item in data['intents']:
    for pattern in item['patterns']:
        all_patterns.append(pattern)  # Flatten patterns
        all_labels.append(item['tag'])

# Adapt TextVectorization to the patterns
vectorize_layer = TextVectorization(
    output_mode='multi_hot',
)
vectorize_layer.adapt(all_patterns)

# Get actual vocabulary size
vocab_size = len(vectorize_layer.get_vocabulary())

# Update the output_sequence_length to match the vocabulary size
vectorize_layer.output_sequence_length = vocab_size

# One-hot encode labels
unique_classes = sorted(set(all_labels))
class_to_index = {label: i for i, label in enumerate(unique_classes)}
num_classes = len(unique_classes)

output_vectors = to_categorical(
    [class_to_index[label] for label in all_labels],
    num_classes=num_classes
)

# Generate multi-hot encoded inputs
multi_hot_inputs = np.array(vectorize_layer(all_patterns).numpy())

# Define the function to optimize


def train_model(learning_rate, batch_size, epochs, dropout_rate):
    model = Sequential()
    model.add(Dense(64, input_shape=(vocab_size,), activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(y_train[0]), activation='softmax'))

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    hist = model.fit(
        multi_hot_inputs, output_vectors,
        epochs=200, batch_size=int(batch_size),
        verbose=0,
        callbacks=[early_stopping, reduce_lr],
        validation_data=(X_test, y_test)
    )
    # Return the last validation accuracy
    return hist.history['val_accuracy'][-1]


# Define the bounds for the hyperparameters
pbounds = {
    'learning_rate': (0.0001, 0.1),  # Adjust the range as necessary
    'batch_size': (8, 64),           # Adjust the range as necessary
    'epochs': (50, 300),             # Adjust the range as necessary
    # Adjust dropout rate range from 0.1 to 0.5
    'dropout_rate': (0.1, 0.5),
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=pbounds,
    random_state=42,
)

# Perform the optimization
optimizer.maximize(init_points=5, n_iter=10)
# Print the best hyperparameters found
print("Best hyperparameters found:")
print(optimizer.max)

# Retrieve the best hyperparameters
best_learning_rate = optimizer.max['params']['learning_rate']
best_batch_size = int(optimizer.max['params']['batch_size'])
best_epochs = int(optimizer.max['params']['epochs'])
best_dropout_rate = optimizer.max['params']['dropout_rate']

# Train the final model with the best hyperparameters
final_model = Sequential()
final_model.add(Dense(64,  input_shape=(vocab_size,), activation='elu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_dropout_rate))
final_model.add(Dense(64, activation='elu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_dropout_rate))
final_model.add(Dense(len(y_train[0]), activation='softmax'))

adam = Adam(learning_rate=best_learning_rate)
final_model.compile(loss='categorical_crossentropy',
                    optimizer=adam, metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# Train the final model
final_hist = final_model.fit(
    multi_hot_inputs, output_vectors,
    epochs=best_epochs, batch_size=best_batch_size,
    verbose=1,
    callbacks=[early_stopping, reduce_lr],
    validation_data=(X_test, y_test)
)

# Save the final model
final_model.save('models/chatbot_model.keras')

# Evaluate the final model
loss, accuracy = final_model.evaluate(
    multi_hot_inputs, output_vectors, verbose=0)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

# Plotting the training and validation accuracy
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(final_hist.history['accuracy'], label='Training Accuracy')
plt.plot(final_hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(final_hist.history['loss'], label='Training Loss')
plt.plot(final_hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Show plots
plt.tight_layout()
plt.show()
