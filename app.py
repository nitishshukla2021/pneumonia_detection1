import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# -----------------------------
# Load Model From Local File
# -----------------------------
@st.cache_resource
def load_pneumonia_model():
    model = load_model("model.h5")
    return model

model = load_pneumonia_model()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_image(image):
    """
    Preprocess uploaded image to match DenseNet121 training.
    - Resize to 224x224
    - Convert to RGB
    - Expand dims and apply DenseNet preprocess_input
    """
    image = image.resize((224, 224))
    image = image.convert("RGB")          # 3 channels
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# -----------------------------
# Initialize Session State
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.pred_score = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Pneumonia Detection using CNN (DenseNet121)")
st.write("Upload a Chest X-ray image to check whether pneumonia is present or not.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Resize for display
    display_image = image.copy()
    display_image.thumbnail((400, 400))
    st.image(display_image, caption="Uploaded X-ray", use_column_width=False)

    # Preprocess and predict
    img = preprocess_image(image)
    pred = model.predict(img)[0]  # shape (2,) because softmax over 2 classes
    pred_class = np.argmax(pred)
    st.session_state.pred_score = pred[pred_class]

    if pred_class == 1:  # assuming index 1 = PNEUMONIA
        st.session_state.result = "PNEUMONIA DETECTED"
    else:
        st.session_state.result = "NORMAL"

# Display prediction
if st.session_state.result:
    st.subheader("🔍 Prediction Result:")
    st.write(f"*Model Output Score:* {st.session_state.pred_score:.4f}")

    if st.session_state.result == "PNEUMONIA DETECTED":
        st.error("⚠ *PNEUMONIA DETECTED*")
        st.markdown(
            """
- The model suggests the presence of pneumonia in the chest X-ray.
- Please consult a medical professional for accurate diagnosis.
- Early detection can help in better treatment and management.
            """
        )
    else:
        st.success("✅ *NORMAL – No Pneumonia Detected*")
        st.markdown(
            """
- The model indicates that the X-ray appears normal.
- No signs of pneumonia were detected.
- If symptoms persist, a medical checkup is still recommended.
            """
        )
