from data import create_dataset, split_data
from model import SimpleNaiveBayes
import numpy as np

def main():
    """
    Main function to run the Naive Bayes example.
    """
    # 1. Create and split the dataset
    X, y = create_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Naive Bayes model...")
    # 2. Initialize and train the model
    model = SimpleNaiveBayes()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # 4. Make predictions on a new sample (example from the article: Sunny, Hot, Normal, False)
    # Note: Our simple model expects categorical inputs as strings in numpy array.
    new_sample = np.array([['Sunny', 'Hot', 'Normal', 'False']])
    prediction = model.predict(new_sample)
    print(f"\nPrediction for {new_sample[0]}: {prediction[0]}")

if __name__ == "__main__":
    main()
