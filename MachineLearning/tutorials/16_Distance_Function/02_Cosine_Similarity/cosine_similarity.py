import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def calculate_cosine_similarity(vector1, vector2):
  """
  Calculates the cosine similarity between two vectors.

  Args:
    vector1: A list or numpy array representing the first vector.
    vector2: A list or numpy array representing the second vector.

  Returns:
    The cosine similarity between the two vectors.
  """
  vector1_np = np.array(vector1)
  vector2_np = np.array(vector2)

  # Compute dot product
  dot_product = np.dot(vector1_np, vector2_np)

  # Compute L2 norms
  norm_v1 = np.linalg.norm(vector1_np)
  norm_v2 = np.linalg.norm(vector2_np)

  # Avoid division by zero if a vector is zero
  if norm_v1 == 0 or norm_v2 == 0:
    return 0.0 # Or handle as an error, depending on desired behavior

  similarity = dot_product / (norm_v1 * norm_v2)
  return similarity

def calculate_cosine_distance_scipy(vector1, vector2):
    """
    Calculates the cosine distance using scipy.spatial.distance.cosine.
    Note: SciPy's cosine distance is 1 - cosine similarity for non-zero vectors.

    Args:
        vector1: A list or numpy array representing the first vector.
        vector2: A list or numpy array representing the second vector.

    Returns:
        The cosine distance between the two vectors.
    """
    return cosine(vector1, vector2)

# Example usage
vector_x = [1, 1, 0, 1, 0, 1]
vector_y = [1, 1, 1, 0, 1, 0]

similarity_manual = calculate_cosine_similarity(vector_x, vector_y)
distance_scipy = calculate_cosine_distance_scipy(vector_x, vector_y)
similarity_scipy = 1 - distance_scipy # Derive similarity from scipy distance

print(f"Vector X: {vector_x}")
print(f"Vector Y: {vector_y}")
print(f"Cosine similarity (manual): {similarity_manual}")
print(f"Cosine similarity (scipy derived): {similarity_scipy}")
print(f"Cosine distance (scipy): {distance_scipy}")

# Example with different vectors
vector_p = [3, 4]
vector_q = [1, 2]

similarity_p_q = calculate_cosine_similarity(vector_p, vector_q)
print(f"\nVector P: {vector_p}")
print(f"Vector Q: {vector_q}")
print(f"Cosine similarity between P and Q: {similarity_p_q}")

# Example with opposite vectors
vector_r = [1, 2]
vector_s = [-1, -2]

similarity_r_s = calculate_cosine_similarity(vector_r, vector_s)
print(f"\nVector R: {vector_r}")
print(f"Vector S: {vector_s}")
print(f"Cosine similarity between R and S: {similarity_r_s}")

# Example: Text Similarity (simplified) using Bag-of-Words
print("\n--- Example: Text Similarity ---")

# Simple sentences
sentence1 = "The sky is blue"
sentence2 = "The sun is bright"
sentence3 = "The sky is bright today"

# Create a vocabulary (unique words)
vocabulary = sorted(list(set(sentence1.lower().split() + sentence2.lower().split() + sentence3.lower().split())))
print(f"Vocabulary: {vocabulary}")

# Function to create a bag-of-words vector
def text_to_vector(text, vocab):
    vector = [0] * len(vocab)
    for word in text.lower().split():
        try:
            idx = vocab.index(word)
            vector[idx] += 1
        except ValueError:
            # Ignore words not in vocabulary
            pass
    return vector

# Convert sentences to vectors
vector_s1 = text_to_vector(sentence1, vocabulary)
vector_s2 = text_to_vector(sentence2, vocabulary)
vector_s3 = text_to_vector(sentence3, vocabulary)

print(f"Vector Sentence 1 ({sentence1}): {vector_s1}")
print(f"Vector Sentence 2 ({sentence2}): {vector_s2}")
print(f"Vector Sentence 3 ({sentence3}): {vector_s3}")

# Calculate cosine similarity between sentences
sim_s1_s2 = calculate_cosine_similarity(vector_s1, vector_s2)
sim_s1_s3 = calculate_cosine_similarity(vector_s1, vector_s3)
sim_s2_s3 = calculate_cosine_similarity(vector_s2, vector_s3)

print(f"\nCosine similarity between Sentence 1 and Sentence 2: {sim_s1_s2}")
print(f"Cosine similarity between Sentence 1 and Sentence 3: {sim_s1_s3}")
print(f"Cosine similarity between Sentence 2 and Sentence 3: {sim_s2_s3}")

# Observe how similarity scores reflect shared vocabulary and indicate semantic closeness to some extent

# Example: Visualization of Cosine Similarity (2D)
print("\n--- Example: Visualization (2D) ---")

# Define vectors
vector_a = np.array([2, 1])
vector_b = np.array([1, 2])
vector_c = np.array([-2, -1])
vector_d = np.array([1, -2])

# Calculate similarities
sim_a_b = calculate_cosine_similarity(vector_a, vector_b)
sim_a_c = calculate_cosine_similarity(vector_a, vector_c)
sim_a_d = calculate_cosine_similarity(vector_a, vector_d)

print(f"Vector A: {vector_a}, Vector B: {vector_b}, Similarity(A, B): {sim_a_b:.2f}")
print(f"Vector A: {vector_a}, Vector C: {vector_c}, Similarity(A, C): {sim_a_c:.2f}")
print(f"Vector A: {vector_a}, Vector D: {vector_d}, Similarity(A, D): {sim_a_d:.2f}")

# Plotting the vectors
plt.figure()
ax = plt.gca()

# Plot vectors from origin
ax.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_key=vector_a, scale=1, color='r', label='Vector A')
ax.quiver(0, 0, vector_b[0], vector_b[1], angles='xy', scale_key=vector_b, scale=1, color='b', label='Vector B')
ax.quiver(0, 0, vector_c[0], vector_c[1], angles='xy', scale_key=vector_c, scale=1, color='g', label='Vector C')
ax.quiver(0, 0, vector_d[0], vector_d[1], angles='xy', scale_key=vector_d, scale=1, color='purple', label='Vector D')

# Set plot limits and aspect ratio
max_val = max(np.max(np.abs(vector_a)), np.max(np.abs(vector_b)), np.max(np.abs(vector_c)), np.max(np.abs(vector_d))) * 1.2
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Cosine Similarity Visualization (2D)')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# Draw origin axes
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)

plt.legend()
plt.show() 