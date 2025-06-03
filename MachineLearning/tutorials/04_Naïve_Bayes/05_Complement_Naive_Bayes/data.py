import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def create_imbalanced_text_dataset():
    """
    Creates a simple imbalanced text dataset for demonstration.
    Class 0 (Majority): Simple positive/neutral sentences.
    Class 1 (Minority): Simple negative sentences.
    """
    texts = [
        "This is a good day",
        "I am happy today",
        "The weather is nice",
        "This is a simple sentence",
        "Machine learning is interesting",
        "Python is a programming language",
        "It is raining outside",
        "Let's go for a walk",
        "I like this",
        "This is great",
        "This is bad",
        "I am sad",
        "This is terrible"
    ]

    labels = [
        "positive/neutral", "positive/neutral", "positive/neutral", "positive/neutral",
        "positive/neutral", "positive/neutral", "positive/neutral", "positive/neutral",
        "positive/neutral", "positive/neutral", # 10 samples for majority class
        "negative", "negative", "negative" # 3 samples for minority class
    ]

    return texts, np.array(labels)

def vectorize_text(texts):
    """
    Converts a list of text documents into a matrix of token counts.
    Uses CountVectorizer.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    return X, vocabulary, vectorizer

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.
    Using stratification to maintain class distribution in test set if possible.
    """
    # Stratify split to help maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    texts, labels = create_imbalanced_text_dataset()
    print("Original texts:", texts)
    print("Original labels:", labels)
    print("Class distribution:", np.unique(labels, return_counts=True))

    X, vocabulary, vectorizer = vectorize_text(texts)
    print("\nVectorized data (X) shape:", X.shape)
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary:\n", vocabulary)

    X_train, X_test, y_train, y_test = split_data(X, labels)
    print("\nDataset split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("\nTraining class distribution:", np.unique(y_train, return_counts=True))
    print("Testing class distribution:", np.unique(y_test, return_counts=True)) 