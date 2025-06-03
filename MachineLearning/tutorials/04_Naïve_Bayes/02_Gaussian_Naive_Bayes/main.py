from data import create_synthetic_dataset, split_data
from model import SimpleGaussianNaiveBayes
import numpy as np

def main():
    """
    Main function to run the Gaussian Naive Bayes example.
    """
    # 1. Create and split the dataset
    X, y = create_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Gaussian Naive Bayes model...")
    # 2. Initialize and train the model
    model = SimpleGaussianNaiveBayes()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # 4. Make predictions on a new sample
    # Example new sample: a point that is likely in class 0 and one likely in class 1
    new_samples = np.array([[2.5, 2.5], [5.5, 5.5]])
    predictions = model.predict(new_samples)
    print(f"\nPredictions for {new_samples[0]}: {int(predictions[0])}")
    print(f"Predictions for {new_samples[1]}: {int(predictions[1])}")

if __name__ == "__main__":
    main()
