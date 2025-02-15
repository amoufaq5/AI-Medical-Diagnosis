import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import fastapi
import uvicorn
from PIL import Image
import numpy as np
import cv2

# Load Pretrained Image Model (CNN for Medical Imaging)
class MedicalCNN(nn.Module):
    def __init__(self):
        super(MedicalCNN, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 5)  # Example: 5 possible conditions
    
    def forward(self, x):
        return self.model(x)

image_model = MedicalCNN()
image_model.load_state_dict(torch.load("medical_cnn.pth"))
image_model.eval()

# Load Pretrained NLP Model for Text Diagnosis
nlp_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# FastAPI for Backend
app = fastapi.FastAPI()

# Image Preprocessing Function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Text Preprocessing Function
def preprocess_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    return inputs

@app.post("/diagnose/image")
async def diagnose_image(file: fastapi.UploadFile):
    image = Image.open(file.file).convert("RGB")
    tensor_image = preprocess_image(image)
    with torch.no_grad():
        prediction = image_model(tensor_image)
    condition_idx = torch.argmax(prediction, dim=1).item()
    return {"diagnosis": f"Condition {condition_idx}"}

@app.post("/diagnose/text")
async def diagnose_text(symptoms: str):
    inputs = preprocess_text(symptoms)
    with torch.no_grad():
        prediction = nlp_model(**inputs).logits
    condition_idx = torch.argmax(prediction, dim=1).item()
    return {"diagnosis": f"Condition {condition_idx}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
