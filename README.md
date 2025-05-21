**Draw-a-Digit – lightweight MNIST**

An interactive sketch-pad that runs a fully-connected neural network written from scratch in NumPy and visualises the pixels that influenced its decision.



Quick start

# Clone the repo
$ git clone https://github.com/<user>/draw-a-digit.git
$ cd draw-a-digit

# Install dependencies
$ pip install -r requirements.txt

# Launch the Streamlit 
$ streamlit run model/app.py

Visit http://localhost:8501, draw any digit, press Predict.

⸻

Project structure

model/
 ├─ app.py              ← Streamlit interface
 ├─ scratch_nn.py       ← NumPy inference helpers
 ├─ mnist_scratch_weights.npz  (≈60 kB)
 └─ __init__.py
requirements.txt         ← numpy, opencv-python-headless, streamlit, drawable-canvas


⸻

Technology
	•	NumPy only – no TensorFlow or PyTorch at inference time.
	•	OpenCV – converts the canvas RGBA image to a 28 × 28 binary mask.
	•	Streamlit – turns a script into a web app with a few lines.

The model achieves ~98 % accuracy on the MNIST test set; free-hand sketches land around 80–85 %.
