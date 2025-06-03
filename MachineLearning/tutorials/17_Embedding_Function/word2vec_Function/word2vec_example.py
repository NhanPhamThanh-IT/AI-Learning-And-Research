import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
import os

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/brown')
except nltk.downloader.DownloadError:
    nltk.download('brown')
except LookupError:
     nltk.download('brown')

# Load the Brown corpus as sample data
# In a real scenario, you would use your own text data
sentences = brown.sents()
print(f"Loaded {len(sentences)} sentences from the Brown corpus.")

# Train the Word2Vec model
# You can adjust parameters like vector_size, window, min_count, workers, sg (for skip-gram or cbow)
# sg=0 for CBOW, sg=1 for Skip-gram (default)

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0) # Example: CBOW model

print("\n--- Word2Vec Model Training Complete ---")

# Build the vocabulary
model.build_vocab(sentences)

# Train the model (if not already trained in initialization)
# model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

# Example Usage:

# 1. Find words similar to a given word
word = 'woman'
print(f"\nWords most similar to '{word}':")
try:
    similar_words = model.wv.most_similar(word)
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
except KeyError:
    print(f"'{word}' not in vocabulary.")

# 2. Perform word analogies
# Example: king - man + woman = ? (should be close to queen)
print("\nWord analogy: king - man + woman = ?")
try:
    analogy_result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    if analogy_result:
        word, similarity = analogy_result[0]
        print(f"Result: '{word}' with similarity {similarity:.4f}")
    else:
        print("Could not find analogy result.")
except KeyError as e:
    print(f"Could not perform analogy. Missing word in vocabulary: {e}")

# 3. Get the vector for a word
word_vector = model.wv['computer']
print(f"\nVector for 'computer' (first 10 elements): {word_vector[:10]}...")

# 4. Calculate similarity between two words
word_a = 'good'
word_b = 'bad'
print(f"\nSimilarity between '{word_a}' and '{word_b}':")
try:
    similarity_a_b = model.wv.similarity(word_a, word_b)
    print(f"{similarity_a_b:.4f}")
except KeyError as e:
    print(f"Could not calculate similarity. Missing word in vocabulary: {e}")

# Note: For visualization of Word2Vec embeddings (e.g., using t-SNE), you would get all word vectors:
# word_vectors = model.wv.vectors
# words = list(model.wv.index_to_key)
# Then apply t-SNE/UMAP and plot.

# Optional: Save and load the model
# model.save("word2vec_brown_cbow.model")
# loaded_model = Word2Vec.load("word2vec_brown_cbow.model") 