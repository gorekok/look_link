# 🔍 LookLink: AI-Powered Visual Search for E-Commerce

[![Status: Work in Progress](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)]()

> **Canlı Demo:** [looklinkai.com](http://looklinkai.com)

LookLink, kullanıcıların beğendikleri bir kombinin fotoğrafını yükleyerek e-ticaret sitelerindeki en benzer ürünleri ve satın alma linklerini saniyeler içinde bulmalarını sağlayan yapay zeka destekli bir görsel arama motorudur.

---

## 🚀 Özellikler (Features)
- **Nesne Tespiti:** Yüklenen fotoğraftaki kıyafetleri (gömlek, pantolon, çanta vb.) otomatik olarak algılar ve kırpar.
- **Derin Öğrenme ile Özellik Çıkarımı:** Kırpılan görsellerin matematiksel vektör haritasını çıkarır.
- **Vektörel Arama:** Saniyeler içinde veritabanındaki binlerce ürün arasından kosinüs benzerliği en yüksek olanları getirir.
- **Kullanıcı Dostu Arayüz:** Streamlit tabanlı, hızlı ve minimalist web arayüzü.

---

## 🛠️ Kullanılan Teknolojiler (Tech Stack)
Bu proje, makine öğrenmesi ve veri madenciliği süreçleri (CRISP-DM) göz önünde bulundurularak aşağıdaki teknolojilerle inşa edilmiştir:

* **Arayüz (Frontend):** Streamlit
* **Nesne Tespiti (Object Detection):** YOLOv8 (Ultralytics)
* **Özellik Çıkarımı (Feature Extraction):** ResNet50 (Keras/TensorFlow)
* **Vektör Veritabanı (Vector Search):** FAISS (Facebook AI Similarity Search)
* **Veri İşleme:** Python, Pandas, NumPy

---

## 🚧 Mevcut Durum ve Yol Haritası (Roadmap)
Bu proje şu an **aktif geliştirme aşamasındadır**. Mimari altyapı (model entegrasyonu ve bulut yayını) tamamlanmış olup, doğruluk oranını artırmak için veri tabanı genişletme çalışmaları devam etmektedir.

- [x] YOLOv8 ve ResNet50 modellerinin entegrasyonu
- [x] FAISS vektör arama altyapısının kurulması
- [x] Streamlit Cloud üzerinden canlıya alma ve özel alan adı bağlantısı
- [ ] **(Güncel Aşama):** Web scraping botları ile e-ticaret (affiliate) ürün verilerinin toplanması ve FAISS indeksinin gerçek verilerle doldurulması.
- [ ] YOLOv8 modelinin spesifik moda veri setleriyle (Fine-tuning) yeniden eğitilmesi.

---

## 💻 Geliştirici (Contact)
Bu proje, veri bilimi ve yapay zeka alanındaki yetenekleri geliştirmek ve sergilemek amacıyla geliştirilmiştir.

* **Geliştirici:** Emir Görkem Yüksel
* **İletişim:** gorkemuni00@gmail.com
