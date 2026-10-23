from ultralytics import YOLO

print("YOLO modeli yükleniyor...")
model = YOLO('models/yolov8n.pt') 

print("Moda görseli analiz ediliyor ve parçalar kesiliyor...")
sonuclar = model('https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?q=80&w=400', save=True, save_crop=True, classes=[0])

print("İşlem tamamlandı! Lütfen sol tarafta oluşan 'runs' klasörünü kontrol et.")