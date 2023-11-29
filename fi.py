
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def preprocess_text_data(text_data):
    # Tokenize and convert text data into sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    
 
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences


def create_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_evaluate_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = create_neural_network(input_dim)
    

    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    
  
    y_pred = model.predict_classes(X_test)
    
  
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    nb_classifier = MultinomialNB()
    
    
    nb_classifier.fit(X_train, y_train)
    
    
    y_pred = nb_classifier.predict(X_test)
    
   
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Function to build and train the classification model
def build_and_train_model(file_path):
    # Load data from the CSV file
    df = pd.read_csv(file_path)
    X = df['text'].values
    y = df['label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    nb_accuracy = train_evaluate_naive_bayes(X_train_vectorized, y_train, X_test_vectorized, y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy}")

    X_train_preprocessed = preprocess_text_data(X_train)
    X_test_preprocessed = preprocess_text_data(X_test)

    y_train_binary = to_categorical(y_train)
    y_test_binary = to_categorical(y_test)


    nn_accuracy = train_evaluate_neural_network(X_train_preprocessed, y_train_binary, X_test_preprocessed, y_test_binary)
    print(f"Neural Network Accuracy: {nn_accuracy}")


    if nn_accuracy > nb_accuracy:
        y_pred_nn = model.predict_classes(X_test_preprocessed)
        display_confusion_matrix(y_test, y_pred_nn)
    else:
        y_pred_nb = nb_classifier.predict(X_test_vectorized)
        display_confusion_matrix(y_test, y_pred_nb)


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        build_and_train_model(file_path)


root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("300x100")


select_button = tk.Button(root, text="Select CSV File", command=select_file)
select_button.pack(pady=20)


root.mainloop()
