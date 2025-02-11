# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values  # All columns except the last as features
    y = data.iloc[:, -1].values  # Last column as target
    return X, y

# Step 2: Split the data into training and testing sets
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Train the KNN model
def train_knn(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_knn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Step 5: Visualize the results (for 2D data only)
def visualize_knn(X_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))

    # Plot true labels
    for i, label in enumerate(np.unique(y_test)):
        plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1],
                    label=f'True Label {label}', marker='o', edgecolor='k', s=100, alpha=0.6)
    
    # Plot predicted labels
    for i, label in enumerate(np.unique(y_pred)):
        plt.scatter(X_test[y_pred == label, 0], X_test[y_pred == label, 1],
                    label=f'Predicted Label {label}', marker='x', s=100, alpha=0.9)
    
    plt.title('KNN Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Main function to run the program
def main():
    # Step 1: Load dataset from CSV file
    csv_file = 'knn_data.csv'  # Replace with your CSV file path
    X, y = load_data(csv_file)
    
    # Step 2: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Step 3: Train the model
    n_neighbors = 3  # Number of neighbors
    model = train_knn(X_train, y_train, n_neighbors=n_neighbors)
    
    # Step 4: Evaluate the model
    evaluate_knn(model, X_test, y_test)
    
    # Step 5: Visualize the results (for 2D data only)
    if X.shape[1] == 2:  # Only visualize if there are 2 features
        y_pred = model.predict(X_test)
        visualize_knn(X_test, y_test, y_pred)

# Run the program
if __name__ == "__main__":
    main()
