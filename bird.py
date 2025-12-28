import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time

model = load_model("final_model.h5")


# App Title

st.title("🐦 Bird vs 🚁 Drone Image Classifier")
st.write("""
Upload an image and let the model tell whether it's a **Bird** or a **Drone**!
""")
# File Uploader
uploaded = st.file_uploader("🔽 Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded:
    # Show image
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Preprocess
    img = image.resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Button
    if st.button("🔍 Predict"):
        with st.spinner("Running model..."):
            pred = model.predict(img_arr)[0][0]
            confidence = pred if pred > 0.5 else 1 - pred
            if pred > 0.5:
                label = "🚁 Drone"
            else:
                label = "🐦 Bird"

        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

            