# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file, encoding='latin1')
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    # Convert text to lowercase
    data['text'] = data['text'].str.lower()
    return data

# Step 3: Split the data into training and testing sets
def split_data(data):
    X = data['text']  # Features (email text)
    y = data['label']  # Labels (spam or ham)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Vectorize the text data
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer(stop_words='english')  # Remove stopwords
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# Step 5: Train the Naive Bayes model
def train_model(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Step 7: Predict spam immediately after user input
def predict_spam(model, vectorizer):
    while True:
        email = input("Enter an email to check if it's spam (or type 'exit' to stop): ")
        if email.lower() == 'exit':
            print("Terminating...")
            break
        email_vec = vectorizer.transform([email])
        prediction = model.predict(email_vec)[0]
        print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}\n")

# Main function to run the program
def main():
    # Step 1: Load dataset from CSV file
    csv_file = 'spam_data.csv'  # Replace with your CSV file path
    data = load_data(csv_file)
    
    # Step 2: Preprocess the data
    data = preprocess_data(data)
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Step 4: Vectorize the text data
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # Step 5: Train the model
    model = train_model(X_train_vec, y_train)
    
    # Step 6: Evaluate the model
    evaluate_model(model, X_test_vec, y_test)
    
    # Step 7: Predict spam immediately after each input
    predict_spam(model, vectorizer)

# Run the program
if __name__ == "__main__":
    main()
