from data import create_spam_dataset, vectorize_text_binary, split_data
from model import SimpleBernoulliNaiveBayes
import numpy as np

def main():
    """
    Main function to run the Bernoulli Naive Bayes example for text classification.
    """
    # 1. Create the dataset
    texts, labels = create_spam_dataset()

    # 2. Vectorize the text data into binary features
    X, vocabulary, vectorizer = vectorize_text_binary(texts)
    print("\nDataset created and binary vectorized.")
    print("Vocabulary size:", len(vocabulary))
    print("Features (X) shape:", X.shape)

    # 3. Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, labels)
    print("\nDataset split into training and testing sets.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # 4. Initialize and train the Bernoulli Naive Bayes model
    print("\nTraining Bernoulli Naive Bayes model...")
    model = SimpleBernoulliNaiveBayes()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")

    # 6. Make a prediction on a new sample (example from the article)
    new_message = ["buy now"]
    # Need to vectorize the new message using the *same* vectorizer fitted on training data
    new_message_vectorized = vectorizer.transform(new_message).toarray()
    prediction = model.predict(new_message_vectorized)
    print(f"\nNew message: '{new_message[0]}'")
    print(f"Predicted class: {prediction[0]}")

if __name__ == "__main__":
    main()
