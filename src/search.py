import numpy as np
import faiss

print("Arama motoru (FAISS) başlatılıyor...")

vektor_boyutu = 1000 
urun_sayisi = 100 

veritabani_vektorleri = np.random.random((urun_sayisi, vektor_boyutu)).astype('float32')

index = faiss.IndexFlatL2(vektor_boyutu)
index.add(veritabani_vektorleri)
print(f"Sisteme {index.ntotal} adet e-ticaret ürünü yüklendi.")

print("Bizim kestiğimiz fotoğrafa en çok benzeyen 3 ürün aranıyor...")
bizim_kiyafetimiz = np.random.random((1, vektor_boyutu)).astype('float32')

mesafeler, urun_id_numaralari = index.search(bizim_kiyafetimiz, 3)

print("\n--- EŞLEŞME BULUNDU ---")
print(f"Veritabanındaki şu ID numaralı ürünler kullanıcının kıyafetine en çok benzeyenlerdir:")
print(urun_id_numaralari[0])