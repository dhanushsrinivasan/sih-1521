import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from lime import lime_tabular

# Load your dataset from the CSV file
data = pd.read_csv(r"/content/FinalBatch (1).csv")

# Assume your dataset has features X and target variable y
X_2d = data.drop('0', axis=1).values  # Assuming '681' is the target column
y = data['1'].values

# Create a LabelEncoder object
le = LabelEncoder()

# Fit the encoder to the unique values in y and transform the target variable
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_2d, y_encoded, test_size=0.2, random_state=42)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Get the number of classes
num_class = len(le.classes_)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=num_class, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model with the encoded target variable
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Evaluate the model
y_pred_prob = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_prob, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Use LIME to explain the prediction
explainer = lime_tabular.LimeTabularExplainer(X_train, random_state=42)

# Choose a sample from the test set for explanation
sample_index = 0
explanation = explainer.explain_instance(X_test[0], model.predict, num_features=len(data.columns) - 1)

# Display LIME explanation
explanation.show_in_notebook()