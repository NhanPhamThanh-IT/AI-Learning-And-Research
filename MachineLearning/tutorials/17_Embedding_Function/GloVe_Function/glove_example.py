import numpy as np
import os

# Define the path to the GloVe file
# You need to download the GloVe vectors first.
# For example, download from: https://nlp.stanford.edu/data/glove.6B.zip
# Extract the zip file and place glove.6B.100d.txt (or similar) in a known location.
glove_file_path = 'path/to/your/downloaded/glove.6B.100d.txt' # !!! REPLACE WITH YOUR ACTUAL PATH !!!

# Check if the glove file exists
if not os.path.exists(glove_file_path):
    print(f"Error: GloVe file not found at {glove_file_path}")
    print("Please download the GloVe vectors and update the 'glove_file_path' variable.")
else:
    # Load GloVe embeddings
    print(f"Loading GloVe embeddings from {glove_file_path}...")
    embeddings_index = {}
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"Found {len(embeddings_index)} word vectors.")

    # Function to get embedding for a word
    def get_word_embedding(word):
        return embeddings_index.get(word.lower())

    # Example Usage:
    word1 = 'king'
    word2 = 'man'
    word3 = 'woman'
    word4 = 'queen'

    vec1 = get_word_embedding(word1)
    vec2 = get_word_embedding(word2)
    vec3 = get_word_embedding(word3)
    vec4 = get_word_embedding(word4)

    if vec1 is not None and vec2 is not None and vec3 is not None and vec4 is not None:
        # Demonstrate word analogy: king - man + woman ≈ queen
        analogy_result = vec1 - vec2 + vec3

        print(f"\nVector for '{word1}': {vec1[:10]}...") # print first 10 elements
        print(f"Vector for '{word2}': {vec2[:10]}...")
        print(f"Vector for '{word3}': {vec3[:10]}...")
        print(f"Vector for '{word4}': {vec4[:10]}...")
        print(f"Result of '{word1}' - '{word2}' + '{word3}': {analogy_result[:10]}...")

        # Find the closest word to the analogy result
        print(f"\nFinding the closest word to '{word1}' - '{word2}' + '{word3}'...")
        min_distance = float('inf')
        closest_word = None

        # Use cosine similarity to find closest word (higher similarity = closer)
        # Or Euclidean distance (lower distance = closer)
        from scipy.spatial.distance import cosine

        # Calculate similarity with the target word 'queen'
        similarity_to_queen = 1 - cosine(analogy_result, vec4)
        print(f"Cosine similarity to '{word4}': {similarity_to_queen:.4f}")

        # Search for the overall closest word (can be slow for large vocabularies)
        # This is for demonstration and might take time
        # In practice, efficient libraries or techniques are used for nearest neighbor search
        print("Searching for overall closest word (may take a while)...")
        for word, vector in embeddings_index.items():
            # Skip the input words
            if word in [word1, word2, word3, word4]:
                continue
            try:
                # Use cosine distance for finding the closest word
                dist = cosine(analogy_result, vector)
                if dist < min_distance:
                    min_distance = dist
                    closest_word = word
            except ValueError: # Handle potential issues with vector dimensions
                continue

        if closest_word:
            print(f"The closest word is '{closest_word}' with cosine distance {min_distance:.4f}")
        else:
             print("Could not find a closest word.")

    else:
        print("Could not find embeddings for one or more example words.") 