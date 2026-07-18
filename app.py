import streamlit as st
from PIL import Image
import sys
import importlib

# Force reload of predict module to ensure latest code
if 'predict' in sys.modules:
    importlib.reload(sys.modules['predict'])

from predict import predict_image
import predict
print(f"[DEBUG] Loading predict from: {predict.__file__}")

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Detection")

st.write("Upload an MRI image to detect the tumor type.")

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    with st.spinner("Predicting..."):

        prediction, confidence, heatmap = predict_image(image)

    st.success(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence:.2f}%")
    
    if heatmap is not None:
        if isinstance(heatmap, dict) and "error" in heatmap:
            st.error(f"❌ GradCAM Generation Failed:\n\n{heatmap['error']}\n\n**Traceback:**\n```\n{heatmap['traceback']}\n```")
        else:
            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(
                heatmap,
                caption="Model Attention",
                use_container_width=True
            )
            
            with st.expander("📊 What do the colors mean?"):
                st.markdown("""
                **Grad-CAM Color Legend:**
                - 🔴 **Red/Warm Colors** → High model attention (most important for prediction)
                - 🟡 **Yellow/Orange** → Medium importance areas
                - 🟢 **Green** → Low-to-medium importance
                - 🔵 **Blue/Cool Colors** → Minimal or no model attention
                
                The heatmap shows which regions of the MRI image the neural network focused on 
                to make its prediction about the tumor type.
                """)
    else:
        st.warning("⚠️ Could not generate Grad-CAM heatmap")