import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F


st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #d2e7d2 0%, #e7f5e7 100%);
    }

    body, html {
        font-family: 'Inter', sans-serif;
    }

    /* remove streamlit's hidden white blocks */
    .stElement,
    .stImage,
    .block-container,
    .stFileUploader,
    .stFileUploader > div {
        background: transparent !important;
    }

    .main-container {
        max-width: 900px;
        margin: auto;
    }

    .hero {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(6px);
        border-radius: 30px;
        padding: 50px 40px;
        margin-top: 40px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.10);
        animation: fadeIn 1s ease;
    }

    .title {
        font-size: 48px;
        font-weight: 600;
        color: #1b3315;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 20px;
        font-weight: 300;
        color: #2d4c28;
        text-align: center;
        margin-bottom: 10px;
    }

    /* upload section */
    .upload-card {
        background: #ffffffee;
        padding: 35px;
        border-radius: 25px;
        margin-top: 40px;
        box-shadow: 0 10px 36px rgba(0,0,0,0.12);
        transition: 0.3s ease;
    }

    .upload-card:hover {
        transform: scale(1.01);
    }

    /* fix uploader label being white */
    .stFileUploader label {
        color: #1b3315 !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    /* result card */
    .result-card {
        background: #ffffffee;
        padding: 40px;
        border-radius: 25px;
        margin-top: 30px;
        box-shadow: 0 10px 36px rgba(0,0,0,0.14);
        animation: fadeIn 0.7s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .pred {
        font-size: 30px;
        font-family: 'Georgia', serif;
        color: #1b3315;
        margin-bottom: 3px;
    }

    .conf {
        font-size: 17px;
        color: #385f35;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #314b2f;
        margin-top: 60px;
        margin-bottom: 20px;
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# HERO BLOCK
# -----------------------------------------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("""
    <div class='hero'>
        <div class='title'>Plant Disease Detector</div>
        <div class='subtitle'>Upload a leaf image to identify possible diseases</div>
    </div>
""", unsafe_allow_html=True)


# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 38)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



# CLASS NAMES
formatted_names = [
    "Apple – Apple Scab", "Apple – Black Rot", "Apple – Cedar Apple Rust", "Apple – Healthy",
    "Blueberry – Healthy", "Cherry – Powdery Mildew", "Cherry – Healthy",
    "Corn – Gray Leaf Spot", "Corn – Common Rust", "Corn – Northern Leaf Blight",
    "Corn – Healthy", "Grape – Black Rot", "Grape – Esca (Black Measles)",
    "Grape – Leaf Blight", "Grape – Healthy", "Orange – Citrus Greening",
    "Peach – Bacterial Spot", "Peach – Healthy", "Pepper – Bacterial Spot",
    "Pepper – Healthy", "Potato – Early Blight", "Potato – Late Blight",
    "Potato – Healthy", "Raspberry – Healthy", "Soybean – Healthy",
    "Squash – Powdery Mildew", "Strawberry – Leaf Scorch", "Strawberry – Healthy",
    "Tomato – Bacterial Spot", "Tomato – Early Blight", "Tomato – Late Blight",
    "Tomato – Leaf Mold", "Tomato – Septoria Leaf Spot", "Tomato – Spider Mites",
    "Tomato – Target Spot", "Tomato – Yellow Leaf Curl Virus",
    "Tomato – Mosaic Virus", "Tomato – Healthy"
]


st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file:
    img = Image.open(uploaded_file)
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        prob = F.softmax(out, dim=1)
        pred = torch.argmax(prob).item()
        confidence = prob[0][pred].item() * 100

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='pred'>{formatted_names[pred]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='conf'>{confidence:.2f}% confidence</div>", unsafe_allow_html=True)

    st.image(img, use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='footer'>Made with Deep Learning · Plant Health Scanner</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
