import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

# =========================
# SIMPLE USER DATABASE (in memory)
# =========================
if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}  # default user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# =========================
# CONFIG
# =========================
IMG_SIZE = 224  # match training size

# =========================
# LOAD MODEL & CLASSES
# =========================
model = load_model("fine_tuned_model.keras")

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

if isinstance(class_names, dict):
    class_names = [k for k,v in sorted(class_names.items(), key=lambda item: item[1])]
if isinstance(class_names, np.ndarray):
    class_names = class_names.tolist()

# =========================
# HELPER FUNCTIONS
# =========================
def preprocess_image(img_file, target_size=(IMG_SIZE, IMG_SIZE)):
    img = Image.open(img_file).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# LOGIN PAGE
# =========================
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Go to Signup"):
        st.session_state.page = "signup"
        st.rerun()

# =========================
# SIGNUP PAGE
# =========================
def signup_page():
    st.title("📝 Signup")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        if new_user in st.session_state.users:
            st.error("Username already exists")
        elif new_user == "" or new_pass == "":
            st.error("Please fill all fields")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created! Please login.")
            st.session_state.page = "login"
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# =========================
# MAIN APP (YOUR ORIGINAL CODE)
# =========================
def main_app():
    st.title("🌾 Rice Leaf Disease Predictor")
    st.write("Upload an image of a rice leaf and the model will predict its disease.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()


    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        img = preprocess_image(uploaded_file)
        pred = model.predict(img)
        top_idx = int(np.argmax(pred[0]))
        confidence = pred[0][top_idx] * 100

        st.write(f"### Prediction: {class_names[top_idx]} ({confidence:.2f}% confidence)")

# =========================
# APP FLOW
# =========================
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
else:
    main_app()
