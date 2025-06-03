import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def create_spam_dataset():
    """
    Creates the simple spam/not spam text dataset from the article.
    """
    data = {
        'text': [
            'buy cheap now',
            'limited offer buy',
            'meet me now',
            'let\'s catch up',
            'Win a new car today!',
            'Lunch plans?',
            'Congratulations! You won a lottery',
            'Can you send me the report?',
            'Exclusive offer for you',
            'Are you coming to the meeting?'
        ],
        'label': ['spam', 'spam', 'not spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
    }
    texts = data['text']
    labels = np.array(data['label'])
    return texts, labels

def vectorize_text_binary(texts):
    """
    Converts a list of text documents into a binary feature matrix.
    Uses CountVectorizer with binary=True.
    """
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    return X, vocabulary, vectorizer

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    texts, labels = create_spam_dataset()
    print("Original texts:", texts)
    print("Original labels:", labels)

    X, vocabulary, vectorizer = vectorize_text_binary(texts)
    print("\nBinary vectorized data (X) shape:", X.shape)
    print("Vocabulary size:", len(vocabulary))
    print("Vocabulary:", vocabulary)

    # Example of binary vectorized data for the first message
    print("\nBinary vectorized first message ('buy cheap now'):", X[0])

    X_train, X_test, y_train, y_test = split_data(X, labels)
    print("\nDataset split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

