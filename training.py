import json
import numpy as np
import nltk
import random
import pickle
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
import tensorflow as tf
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
intents = []

# Load the JSON files
json_files = ['Anaerobic_respiration.json', 'Aerobic_respiration.json',
              'Gas_exchage.json', 'Greetings.json',
              'Response_to_exercise.json', 'Type_of_respiration.json']
for file in json_files:
    intents.append(json.loads(open(file).read()))


# Extract patterns and labels
all_patterns = []
all_labels = []

for json_object in intents:
    for intent in json_object['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)  # Flatten patterns
            all_labels.append(intent['tag'])

classes = sorted(set(all_labels))
pickle.dump(classes, open('models/classes.pkl', 'wb'))


# One-hot encode labels
unique_classes = sorted(set(all_labels))
class_to_index = {label: i for i, label in enumerate(unique_classes)}
num_classes = len(unique_classes)

output_vectors = to_categorical(
    [class_to_index[label] for label in all_labels],
    num_classes=num_classes
)

# Create a TextVectorization layer
vectorize_layer = TextVectorization(output_mode='multi_hot')
vectorize_layer.adapt(all_patterns)

# Create a tf.data.Dataset for training - works well for large datasets
# train_size = 0.7
# dataset = tf.data.Dataset.from_tensor_slices(
#     # tf.constant ensures a 1d array reshape
#     (tf.constant(all_patterns), output_vectors))
# dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# dataset = dataset.shuffle(len(all_labels)).batch(32)
# train = dataset.take(int(len(all_labels) * train_size)
#                      ).batch(32).prefetch(tf.data.AUTOTUNE)
# test = dataset.skip(int(len(all_labels) * train_size)
#                     ).batch(32).prefetch(tf.data.AUTOTUNE)
# test = tf.data.Dataset.from_tensors(test)
dataset = tf.data.Dataset.from_tensor_slices((all_patterns, output_vectors))
dataset = dataset.shuffle(buffer_size=len(all_patterns))

# Split the dataset
dataset_size = len(all_patterns)
train_size = int(0.6 * dataset_size)

train = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
test = dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)


def train_model(learning_rate, batch_size, epochs, dropout_rate):
    model = Sequential()
    model.add(vectorize_layer)  # Vectorize the input text
    model.add(Dense(64, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    hist = model.fit(
        train,
        epochs=200, batch_size=int(batch_size),
        verbose=0,
        callbacks=[early_stopping, reduce_lr],
        validation_data=test
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
final_model.add(vectorize_layer)
final_model.add(Dense(64, activation='elu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_dropout_rate))
final_model.add(Dense(64, activation='elu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_dropout_rate))
final_model.add(Dense(num_classes, activation='softmax'))

adam = Adam(learning_rate=best_learning_rate)
final_model.compile(loss='categorical_crossentropy',
                    optimizer=adam, metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# Train the final model
final_hist = final_model.fit(
    train,
    epochs=best_epochs, batch_size=best_batch_size,
    verbose=1,
    callbacks=[early_stopping, reduce_lr],
    validation_data=test
)

# Save the final model
final_model.save('models/chatbot_model2')

# Evaluate the final model
loss, accuracy = final_model.evaluate(test, verbose=0)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

print(final_model.summary())

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
