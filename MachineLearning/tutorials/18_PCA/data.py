from sklearn.datasets import load_iris
import numpy as np

def load_data():
    data = load_iris()
    return data.data 