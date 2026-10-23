import torch
from torchvision import models, transforms
from PIL import Image

print("ResNet50 yapay zeka modeli yükleniyor...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Kesilen görsel DNA'sına (vektörlere) dönüştürülüyor...")
img_path = "runs/detect/predict-2/crops/person/image0.jpg"
img = Image.open(img_path).convert('RGB')

img_tensor = preprocess(img)
batch_tensor = torch.unsqueeze(img_tensor, 0)

with torch.no_grad():
    vektorler = model(batch_tensor)

print("İşlem tamamlandı! Fotoğrafın yapay zeka dilindeki karşılığı (İlk 5 değer):")
print(vektorler[0][:5])