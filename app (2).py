import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Page config
st.set_page_config(
    page_title="😷 Smart Face Mask Detection",
    layout="wide",
    page_icon="😷"
)

# Load model with caching
@st.cache_resource
def load_model_cached():
    return load_model("project_face_mask_detection.keras")

model = load_model_cached()

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Sidebar
with st.sidebar:
    st.title("🧠 About This App")
    st.markdown("""
    This app uses deep learning to detect whether a person is wearing a face mask.
    
    - Upload or capture an image.
    - Get instant feedback.
    - Built with Streamlit & Keras.
    """)
    st.info("Tip: Use well-lit images with clear faces for best results.")
    st.markdown("---")
    st.caption("📍 Developed by Surendra • 2025")

# Optional resize function (only for uploads)
def resize_image(image, max_size=(400, 400)):
    image = image.copy()
    image.thumbnail(max_size)  # maintains aspect ratio
    return image

# Detection function
def detect_and_predict(image_input):
    image_np = np.array(image_input.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return image_input, None, "⚠️ No face detected"

    x, y, w, h = faces[0]
    face_roi = image_np[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_roi).resize((200, 200))
    img_array = img_to_array(face_pil) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    label = "✅ Mask Detected" if prediction < 0.5 else "🚫 No Mask Detected"

    # Draw results on original image
    color = (0, 255, 0) if prediction < 0.5 else (255, 0, 0)
    cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image_np, f"{label} ({confidence*100:.2f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return Image.fromarray(image_np), confidence, label

# App header
st.markdown("<h1 style='text-align: center;'>😷 AI Face Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload or capture an image to analyze mask presence.</p>", unsafe_allow_html=True)

# Input choice
input_choice = st.selectbox("Choose Input Method", ["📤 Upload Image", "📷 Use Webcam"])

# === Upload Image ===
if input_choice == "📤 Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_input = Image.open(uploaded_file)
        resized_display = resize_image(image_input)

        col1, col2 = st.columns(2)

        with col1:
            st.image(resized_display, caption="Uploaded Image")

        with st.spinner("Analyzing with AI model..."):
            result_img, confidence, label = detect_and_predict(image_input)

        with col2:
            st.image(result_img, caption="Detection Output")
            if confidence is not None:
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
                # st.success(label) if "Mask" in label else st.error(label)
                if "Mask" in label:
                    st.success(label)
                else:
                    st.error(label)
            else:
                st.warning(label)

# === Webcam Input ===
elif input_choice == "📷 Use Webcam":
    col1, col2 = st.columns([1, 3])

    with col1:
        camera_image = st.camera_input("Take a picture using webcam")

    if camera_image:
        image_input = Image.open(camera_image)

        with st.spinner("Analyzing..."):
            result_img, confidence, label = detect_and_predict(image_input)

        with col2:
            st.write("Resulted Image")
            st.image(result_img, caption="Detection Output")
            if confidence is not None:
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
                # st.success(label) if "Mask" in label else st.error(label)
                if "Mask" in label:
                    st.success(label)
                else:
                    st.error(label)

            else:
                st.warning(label)
