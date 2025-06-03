from data import create_imbalanced_text_dataset, vectorize_text, split_data
from model import SimpleComplementNaiveBayes
import numpy as np

def main():
    """
    Main function to run the Complement Naive Bayes example for text classification.
    """
    # 1. Create the dataset
    texts, labels = create_imbalanced_text_dataset()

    # 2. Vectorize the text data
    X, vocabulary, vectorizer = vectorize_text(texts)
    print("\nDataset created and vectorized.")
    print("Vocabulary size:", len(vocabulary))
    print("Features (X) shape:", X.shape)

    # 3. Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, labels)
    print("\nDataset split into training and testing sets.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Training class distribution:", np.unique(y_train, return_counts=True))
    print("Testing class distribution:", np.unique(y_test, return_counts=True))

    # 4. Initialize and train the Complement Naive Bayes model
    print("\nTraining Complement Naive Bayes model...")
    model = SimpleComplementNaiveBayes()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # 6. Make predictions on new samples
    new_texts = ["This is a very good day", "I am feeling very terrible", "a neutral sentence"]
    new_X = vectorizer.transform(new_texts).toarray()

    predictions = model.predict(new_X)

    print("\nPredictions on new samples:")
    for text, pred in zip(new_texts, predictions):
        print(f"Message: '{text}' -> Predicted Class: {pred}")

if __name__ == "__main__":
    main() 