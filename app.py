
import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import MyModel  # تأكد من أن لديك تعريف الموديل في ملف `model.py`

# إعداد الجهاز
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# تحميل الموديل
@st.cache_resource
def load_model(model_path, num_classes=5):
    model = MyModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model_path = "model_26"  # ارفع هذا الملف مع المشروع
model = load_model(model_path)

# تعريف التحويلات
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# واجهة التطبيق
st.title("Brain Tumor Classification")
st.text("Upload an MRI image to classify the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تطبيق التحويلات على الصورة
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # توقع الفئة
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # فئات التصنيف
    labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    st.write(f"**Prediction:** {labels[predicted.item()]}")
