# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import time
import os

# ======================
# Конфигурация
# ======================
DEVICE = torch.device("cpu")  # Streamlit Cloud не поддерживает GPU

# Классы EuroSAT
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# ======================
# Загрузка модели
# ======================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    # Убедитесь, что модель лежит в той же папке, что и app.py
    model.load_state_dict(torch.load("eurosat_resnet18.pth", map_location=DEVICE))
    model.eval()
    return model

# ======================
# Трансформации
# ======================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ======================
# Функция предсказания
# ======================
def predict(image: Image.Image):
    model = load_model()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = prob[0][pred_idx].item()
    elapsed = time.time() - start
    return EUROSAT_CLASSES[pred_idx], confidence, elapsed

# ======================
# Загрузка изображения
# ======================
def load_image_from_upload(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Не удалось загрузить изображение: {e}")
        return None

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="EuroSAT Classifier", layout="centered")
st.title("🛰️ Классификатор спутниковых снимков (EuroSAT)")
st.markdown("Загрузите изображение — и модель определит тип ландшафта!")

# Загрузка
uploaded_files = st.file_uploader(
    "Выберите изображение (JPG/PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)
url = st.text_input("Или вставьте URL изображения")

# Обработка
images = []

if uploaded_files:
    for f in uploaded_files:
        images.append(load_image_from_upload(f))

if url:
    img = load_image_from_url(url)
    if img:
        images.append(img)

# Вывод результатов
for img in images:
    st.image(img, caption="Загруженное изображение", width=300)
    try:
        pred_class, conf, t = predict(img)
        st.success(f"**Предсказание**: `{pred_class}`")
        st.info(f"**Уверенность**: {conf:.2%} | **Время обработки**: {t:.3f} сек")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
    st.divider()

# Информация
st.sidebar.markdown("### 📌 О проекте")
st.sidebar.markdown("""
- **Датасет**: EuroSAT (27 000 спутниковых снимков)
- **Модель**: ResNet-18 (предобученная, fine-tuning)
- **Классов**: 10 (лес, река, город и др.)
- **Точность**: >95% на валидации
""")