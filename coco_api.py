from flask import Flask, request, jsonify
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

# COCO category mapping (10 classes used in training)
COCO_CATEGORIES = {
    1: "person", 2: "bicycle", 3: "car", 4: "bird", 5: "cat",
    6: "dog", 7: "horse", 8: "sheep", 9: "cow", 10: "bottle"
}

# Google Drive file ID
GDRIVE_FILE_ID = "1460EuNDZswISp85pB0RHRP0zHifv4kiI"

# Load model
def create_model(num_classes=11):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Download model from Google Drive if not present
if not os.path.exists('coco_object_detector.pth'):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, 'coco_object_detector.pth', quiet=False)
    print("✓ Model downloaded!")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=11)

try:
    model.load_state_dict(torch.load('coco_object_detector.pth', map_location=device))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    print("Using default pre-trained weights")

model.to(device)
model.eval()

# Image preprocessing
def get_transform():
    transforms = []
    transforms.append(T.Resize(size=(600, 600)))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

transform = get_transform()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "COCO Object Detection API",
        "model": "Faster R-CNN with ResNet50-FPN",
        "classes": list(COCO_CATEGORIES.values()),
        "endpoints": {
            "/predict": "POST - Upload image for object detection",
            "/health": "GET - Check service health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": str(device)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Read image
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get confidence threshold (default 0.5)
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Preprocess image
        img_tensor = torchvision.io.decode_image(
            torch.frombuffer(image_bytes, dtype=torch.uint8)
        )
        
        if img_tensor.shape[0] == 1:  # Grayscale
            img_tensor = img_tensor.repeat(3, 1, 1)
        
        img_tensor, _ = transform(img_tensor, {})
        
        # Run inference
        with torch.no_grad():
            predictions = model([img_tensor.to(device)])[0]
        
        # Filter predictions by confidence
        mask = predictions['scores'] > confidence_threshold
        boxes = predictions['boxes'][mask].cpu().numpy()
        labels = predictions['labels'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()
        
        # Format results
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                "class": COCO_CATEGORIES.get(int(label), f"class_{label}"),
                "confidence": float(score),
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3])
                }
            })
        
        return jsonify({
            "success": True,
            "num_detections": len(detections),
            "detections": detections,
            "image_size": {"width": image.width, "height": image.height}
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Key changes:**
1. ✅ Added `import os` and `import gdown`
2. ✅ Added your Google Drive file ID: `1460EuNDZswISp85pB0RHRP0zHifv4kiI`
3. ✅ Added code to download model from Google Drive on startup

**Now also update your `requirements.txt`:**
```
Flask==3.0.0
torch==2.1.0
torchvision==0.16.0
Pillow==10.1.0
gunicorn==21.2.0
gdown==4.7.1