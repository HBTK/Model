from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

app = Flask(__name__)

# Load model, tokenizer, and label encoder
MODEL_DIR = "./saved_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.to(device)

with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    texts = data.get("texts", [])
    
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "No valid list of texts provided"}), 400

    results = []
    for text in texts:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).cpu().item()
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        results.append({"text": text, "predicted_class": predicted_class})

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
