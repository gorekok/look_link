import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np
import faiss
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="LookLink | AI Visual Search", layout="wide")

# --- ÖZEL CSS TASARIMI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600&display=swap');
    
    .stApp, .main, .block-container {
        background-color: #FAFAFA !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Üst Bar Tasarımı */
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #262730; 
        color: #FFFFFF;
        text-align: center;
        padding: 15px 0;
        z-index: 999999;
        font-family: 'Fredoka', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 4px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    
    .block-container {
        padding-top: 120px !important;
    }
    
    /* Buton Stilleri */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0px !important;
        border: 2px solid #000000 !important;
        padding: 12px 24px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    .stLinkButton>a {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        text-align: center;
        text-transform: uppercase;
        font-weight: bold;
        transition: all 0.3s ease;
        display: block;
        width: 100%;
    }
    .stLinkButton>a:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    img { border-radius: 0px !important; }
    </style>
    
    <div class="top-bar">LOOKLINK</div>
""", unsafe_allow_html=True)

# --- YAPAY ZEKA MODELLERİ ---
@st.cache_resource
def load_yolo(): return YOLO('models/yolov8n.pt')

@st.cache_resource
def load_resnet():
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.eval()
    return m

@st.cache_resource
def create_dummy_database():
    idx = faiss.IndexFlatL2(1000)
    idx.add(np.random.random((100, 1000)).astype('float32'))
    return idx

def get_product_details(product_id):
    categories = ["Süet Erkek Bot", "Slim Fit Kumaş Pantolon", "Klasik Erkek Gömlek", "Kışlık Kaşe Kaban", "Günlük Spor Ayakkabı"]
    prices = ["2.499,90 ₺", "899,99 ₺", "549,50 ₺", "3.299,00 ₺", "1.899,00 ₺"]
    images = [
        "https://images.unsplash.com/photo-1617114919297-3c8ddb01f599?w=400&q=80",
        "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400&q=80",
        "https://images.unsplash.com/photo-1603252109303-2751441dd157?w=400&q=80",
        "https://images.unsplash.com/photo-1507680434267-2856248f5727?w=400&q=80",
        "https://images.unsplash.com/photo-1520975954732-57dd22299614?w=400&q=80"
    ]
    return {"name": categories[product_id % 5], "price": prices[product_id % 5], "image": images[product_id % 5], "url": f"https://www.trendyol.com/sr?q=erkek+giyim"}

yolo_model = load_yolo()
resnet_model = load_resnet()
faiss_index = create_dummy_database()
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# --- ARAYÜZ MANTIĞI ---

yuklenen_gorsel = st.file_uploader("", type=["jpg", "png", "jpeg"], help="Kombin fotoğrafı yükleyin")

if yuklenen_gorsel is None:
    # Kullanıcı fotoğraf yüklemediğinde senin özel tasarımın görünür
    gorsel_yolu = "sitegorseli.jpeg"
    if os.path.exists(gorsel_yolu):
        st.image(gorsel_yolu, use_container_width=True)
    else:
        st.warning(f"Lütfen '{gorsel_yolu}' dosyasını LookLink klasörünün içine kopyalayın.")
else:
    # Fotoğraf Yüklendikten Sonraki Vitrin Ekranı
    image = Image.open(yuklenen_gorsel).convert('RGB')
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("BENZER ÜRÜNLERİ BUL"):
            with st.spinner("Koleksiyon taranıyor..."):
                sonuclar = yolo_model(image, classes=[0])
                if len(sonuclar[0].boxes) > 0:
                    kutu = sonuclar[0].boxes[0].xyxy[0].tolist() 
                    kes_gorsel = image.crop((kutu[0], kutu[1], kutu[2], kutu[3]))
                    
                    with torch.no_grad():
                        vec = resnet_model(torch.unsqueeze(preprocess(kes_gorsel), 0)).numpy()
                        
                    mesafeler, urun_id_numaralari = faiss_index.search(vec, 3)
                    
                    st.markdown("<h3 style='color:#000;'>ÖNERİLEN PARÇALAR</h3>", unsafe_allow_html=True)
                    urun_kolonlari = st.columns(3)
                    for i, u_id in enumerate(urun_id_numaralari[0]):
                        urun = get_product_details(u_id)
                        with urun_kolonlari[i]:
                            st.image(urun["image"], use_container_width=True)
                            st.markdown(f"<p style='font-size:14px; font-weight:bold; margin-bottom:0px; color:#000;'>{urun['name']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='font-size:14px; color:#555;'>{urun['price']}</p>", unsafe_allow_html=True)
                            st.link_button("İNCELE", urun["url"])
                else:
                    st.error("Görselde analiz edilecek bir kombin bulunamadı.")