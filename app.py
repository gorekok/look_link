import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np
import faiss

@st.cache_resource
def load_yolo():
    return YOLO('models/yolov8n.pt')

@st.cache_resource
def load_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

@st.cache_resource
def create_dummy_database():
    db_vectors = np.random.random((100, 1000)).astype('float32')
    index = faiss.IndexFlatL2(1000)
    index.add(db_vectors)
    return index

# Sahte E-Ticaret Ürün Kataloğu (Simülasyon)
def get_product_details(product_id):
    categories = ["Süet Erkek Bot", "Slim Fit Kumaş Pantolon", "Klasik Erkek Gömlek", "Kışlık Kaşe Kaban", "Günlük Spor Ayakkabı"]
    prices = ["2.499,90 ₺", "899,99 ₺", "549,50 ₺", "3.299,00 ₺", "1.899,00 ₺"]
    images = [
        "https://images.unsplash.com/photo-1617114919297-3c8ddb01f599?w=300&q=80",
        "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300&q=80",
        "https://images.unsplash.com/photo-1603252109303-2751441dd157?w=300&q=80",
        "https://images.unsplash.com/photo-1507680434267-2856248f5727?w=300&q=80",
        "https://images.unsplash.com/photo-1520975954732-57dd22299614?w=300&q=80"
    ]
    
    return {
        "name": f"{categories[product_id % 5]}",
        "price": prices[product_id % 5],
        "image": images[product_id % 5],
        "url": f"https://www.trendyol.com/sr?q=erkek+giyim&ref=looklink_affiliate_{product_id}" # Satış ortaklığı link simülasyonu
    }

yolo_model = load_yolo()
resnet_model = load_resnet()
faiss_index = create_dummy_database()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="LookLink AI", layout="wide")

st.title("👗 LookLink: Görselden Ürün Bulma")
st.write("Sosyal medyada beğendiğiniz bir kombini yükleyin, yapay zeka benzer e-ticaret ürünlerini bulsun!")

yuklenen_gorsel = st.file_uploader("Kombin fotoğrafı yükleyin (JPG, PNG)", type=["jpg", "png", "jpeg"])

if yuklenen_gorsel is not None:
    image = Image.open(yuklenen_gorsel).convert('RGB')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)
    
    with col2:
        if st.button("Benzer Ürünleri Bul", use_container_width=True):
            with st.spinner("Yapay zeka tüm analizleri gerçekleştiriyor..."):
                sonuclar = yolo_model(image, classes=[0])
                
                if len(sonuclar[0].boxes) > 0:
                    kutu = sonuclar[0].boxes[0].xyxy[0].tolist() 
                    kesilmis_gorsel = image.crop((kutu[0], kutu[1], kutu[2], kutu[3]))
                    
                    img_tensor = preprocess(kesilmis_gorsel)
                    batch_tensor = torch.unsqueeze(img_tensor, 0)
                    
                    with torch.no_grad():
                        vektor = resnet_model(batch_tensor).numpy()
                    
                    mesafeler, urun_id_numaralari = faiss_index.search(vektor, 3)
                    
                    st.success("🎉 Eşleşen Ürünler Bulundu!")
                    
                    # Ürünleri yan yana 3 sütun halinde gösterme
                    urun_kolonlari = st.columns(3)
                    
                    for i, u_id in enumerate(urun_id_numaralari[0]):
                        urun = get_product_details(u_id)
                        with urun_kolonlari[i]:
                            st.image(urun["image"], use_container_width=True)
                            st.markdown(f"**{urun['name']}**")
                            st.write(f"🏷️ {urun['price']}")
                            st.link_button("Satın Al 🛒", urun["url"], use_container_width=True)
                else:
                    st.error("Görselde kişi tespit edilemedi.")