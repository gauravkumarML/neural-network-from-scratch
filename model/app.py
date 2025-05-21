

import sys, pathlib
here = pathlib.Path(__file__).resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parent.parent))
import numpy as np, cv2, streamlit as st
from streamlit_drawable_canvas import st_canvas
from scratch_nn import load_params, predict_one

st.set_page_config(page_title="Draw-a-Digit", page_icon="✏️")
st.title("Sketch a digit – The Model will guess!")






canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=12,
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict") and canvas.image_data is not None:
    rgba = canvas.image_data.astype("uint8")
    alpha = rgba[..., 3]
    alpha = cv2.resize(alpha, (28, 28), interpolation=cv2.INTER_NEAREST)
    img = (alpha > 0).astype("float32")

    p = load_params()
    probs = predict_one(img, p)
    pred  = int(np.argmax(probs))

    st.subheader(f"I’m {probs[pred]*100:.1f}% sure this is a **{pred}**")
    st.bar_chart(probs)

    # simple saliency via backward-prop
    if st.checkbox("Show saliency"):
        W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
        x = img.reshape(1, -1)
        z1 = x @ W1 + b1;       a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2;      y = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
        dz2 = y.copy();         dz2[0, pred] -= 1
        da1 = dz2 @ W2.T;       dz1 = da1 * (z1 > 0)
        dx  = dz1 @ W1.T
        sal = np.abs(dx).reshape(28, 28)
        sal /= sal.max()
        st.image(sal, caption="Saliency map", clamp=True, channels="GRAY")
