# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 0].values.reshape(-1, 1)  # First column as feature
    y = data.iloc[:, 1].values  # Second column as target
    return X, y

# Step 2: Split the data into training and testing sets
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Train the Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 4: Make predictions using the trained model
def make_predictions(model, X):
    return model.predict(X)

# Step 5: Evaluate the model
def evaluate_model(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} - Mean Squared Error: {mse:.2f}")
    print(f"{label} - R^2 Score: {r2:.2f}")
    return r2

# Step 6: Visualize the results
def visualize_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, train_accuracy, test_accuracy):
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.plot(X_train, y_train_pred, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Training Data (Accuracy: {train_accuracy:.2%})')
    plt.legend()
    
    # Plot testing data
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, color='green', label='Testing data')
    plt.plot(X_test, y_test_pred, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Testing Data (Accuracy: {test_accuracy:.2%})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main function to run the program
def main():
    # Step 1: Load dataset from CSV file
    csv_file = 'simple_linear_regression_data.csv'  # Replace with your CSV file path
    X, y = load_data(csv_file)
    
    # Step 2: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Make predictions
    y_train_pred = make_predictions(model, X_train)
    y_test_pred = make_predictions(model, X_test)
    
    # Step 5: Evaluate the model
    train_accuracy = evaluate_model(y_train, y_train_pred, "Training")
    test_accuracy = evaluate_model(y_test, y_test_pred, "Testing")
    
    # Step 6: Visualize the results
    visualize_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, train_accuracy, test_accuracy)

# Run the program
if __name__ == "__main__":
    main()
