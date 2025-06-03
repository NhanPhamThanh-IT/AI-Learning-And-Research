import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def calculate_euclidean_distance(point1, point2):
  """
  Calculates the Euclidean distance between two points.

  Args:
    point1: A list or numpy array representing the first point.
    point2: A list or numpy array representing the second point.

  Returns:
    The Euclidean distance between the two points.
  """
  # Using numpy for calculation
  point1_np = np.array(point1)
  point2_np = np.array(point2)
  distance = np.sqrt(np.sum((point1_np - point2_np)**2))
  return distance

def calculate_euclidean_distance_scipy(point1, point2):
  """
  Calculates the Euclidean distance using scipy.spatial.distance.euclidean.

  Args:
    point1: A list or numpy array representing the first point.
    point2: A list or numpy array representing the second point.

  Returns:
    The Euclidean distance between the two points.
  """
  return euclidean(point1, point2)

# Example usage
point_a = [1, 2, 3]
point_b = [4, 5, 6]

distance_manual = calculate_euclidean_distance(point_a, point_b)
distance_scipy = calculate_euclidean_distance_scipy(point_a, point_b)

print(f"Point A: {point_a}")
print(f"Point B: {point_b}")
print(f"Euclidean distance (manual): {distance_manual}")
print(f"Euclidean distance (scipy): {distance_scipy}")

# Example with higher dimensions
point_c = [10, 5, 1, 8]
point_d = [2, 8, 6, 3]

distance_c_d = calculate_euclidean_distance(point_c, point_d)
print(f"\nPoint C: {point_c}")
print(f"Point D: {point_d}")
print(f"Euclidean distance between C and D: {distance_c_d}")

# Example with feature scaling
print("\n--- Example with Feature Scaling ---")
data_points = np.array([[100, 0.1], [200, 0.5], [150, 0.3]]) # Example data with different scales

# Calculate distance without scaling
distance_unscaled = calculate_euclidean_distance(data_points[0], data_points[1])
print(f"Distance without scaling between {data_points[0]} and {data_points[1]}: {distance_unscaled}")

# Apply Min-Max Scaling
scaler = MinMaxScaler()
data_points_scaled = scaler.fit_transform(data_points)

# Calculate distance with scaling
distance_scaled = calculate_euclidean_distance(data_points_scaled[0], data_points_scaled[1])

print(f"Scaled data points:\n{data_points_scaled}")
print(f"Distance with scaling between {data_points_scaled[0]} and {data_points_scaled[1]}: {distance_scaled}")

# Observe how scaling affects the distance calculation when features have different magnitudes

# Example: Visualization of Euclidean Distance (2D)
print("\n--- Example: Visualization (2D) ---")

point_p = np.array([1, 3])
point_q = np.array([4, 1])

distance_p_q = calculate_euclidean_distance(point_p, point_q)

plt.figure()
plt.plot(point_p[0], point_p[1], 'o', label='Point P')
plt.plot(point_q[0], point_q[1], 'o', label='Point Q')

# Draw the line between points P and Q
plt.plot([point_p[0], point_q[0]], [point_p[1], point_q[1]], 'k--')

plt.title(f'Euclidean Distance = {distance_p_q:.2f}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.axis('equal') # Ensures equal scaling for both axes
plt.legend()
plt.show() 