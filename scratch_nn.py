import numpy as np
from pathlib import Path

def load_params(path: str = "model/mnist_scratch_weights.npz"):
    """Load saved NumPy weights and return as a dict."""
    data = np.load(Path(path), allow_pickle=True)
    return {k: data[k] for k in data.files}

# ---------- simple 2‑layer network forward pass ----------
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def predict(x: np.ndarray, p: dict) -> np.ndarray:
    """
    x : (n, 784)  flattened, normalised images
    p : dict of weight matrices/ bias vectors
    returns class‑probabilities (n, 10)
    """
    z1 = x @ p["W1"] + p["b1"]

    a1 = relu(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    return softmax(z2)

def predict_one(img28x28: np.ndarray, p: dict) -> np.ndarray:
    """Convenience wrapper for a single 28×28 image."""
    return predict(img28x28.reshape(1, -1), p)[0]
