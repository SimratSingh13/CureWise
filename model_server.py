"""
Standalone Model Server - Runs in separate process
Enhanced with confidence-based prediction thresholds AND Grad-CAM explainability
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import threading
import time
import cv2
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

model = None
model_loaded = False
loading = False
class_names = ['Normal', 'Pneumonia', 'COVID']
img_size = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'heatmaps')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

MIN_CONFIDENCE = 60.0
UNCERTAIN_THRESHOLD = 15.0
HIGH_RISK_THRESHOLD = 85.0
MEDIUM_RISK_THRESHOLD = 60.0

def load_model():
    global model, model_loaded, loading
    loading = True
    try:
        model_path = os.path.join(BASE_DIR, 'models', 'lung_xray_model.h5')
        print(f"📦 Loading model from {model_path}...")
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile()
        
        # Build model with dummy input
        print("🔧 Building model with dummy input...")
        dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        print("✅ Model built successfully")
        
        # Print model structure
        print("\n📋 Model structure:")
        for i, layer in enumerate(model.layers):
            print(f"   {i}: {layer.name} - {layer.__class__.__name__}")
            if 'mobilenetv2' in layer.name.lower():
                print(f"       Internal Conv2D layers:")
                for sublayer in layer.layers:
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        print(f"          - {sublayer.name}")
        
        model_loaded = True
        print("\n✅ Model loaded and ready for Grad-CAM")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loading = False

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array.astype(np.float32)
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

def generate_gradcam(img_array, class_idx):
    """Generate Grad-CAM heatmap (FINAL FIXED VERSION)"""
    global model
    
    try:
        # ✅ Step 1: Get MobileNetV2 base model
        base_model = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            print("⚠️ MobileNetV2 not found")
            return None

        # ✅ Step 2: Get last Conv layer (Conv_1)
        last_conv_layer = base_model.get_layer("Conv_1")
        print(f"🔥 Using conv layer: {last_conv_layer.name}")

        # ✅ Step 3: Create Grad Model from BASE MODEL (IMPORTANT)
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        # ✅ Step 4: Forward pass through base model
        with tf.GradientTape() as tape:
            conv_outputs, base_predictions = grad_model(img_array)
            
            # Pass base output through top layers manually
            x = base_predictions
            for layer in model.layers[1:]:
                x = layer(x)
            
            predictions = x
            loss = predictions[:, class_idx]

        # ✅ Step 5: Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("⚠️ Gradients are None")
            return None

        # ✅ Step 6: Grad-CAM logic
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        print("✅ Heatmap generated successfully")
        return heatmap.numpy()

    except Exception as e:
        print(f"❌ Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def overlay_heatmap(original_img_path, heatmap, prediction, confidence):
    try:
        img = cv2.imread(original_img_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        text = f"{prediction} ({confidence:.1f}%)"
        cv2.putText(superimposed, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timestamp = int(time.time() * 1000)
        filename = f"heatmap_{timestamp}.jpg"
        path = os.path.join(STATIC_DIR, filename)
        cv2.imwrite(path, superimposed)
        
        return f"/static/heatmaps/{filename}"
        
    except Exception as e:
        print(f"❌ Heatmap overlay failed: {e}")
        return None

def make_smart_prediction(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    top1_prob = predictions[sorted_indices[0]] * 100
    top2_prob = predictions[sorted_indices[1]] * 100
    
    top1_class = class_names[sorted_indices[0]]
    top2_class = class_names[sorted_indices[1]]
    prob_gap = top1_prob - top2_prob
    
    probabilities = {class_names[i]: float(predictions[i]) * 100 for i in range(len(class_names))}
    class_idx = sorted_indices[0]
    
    if top1_prob >= MIN_CONFIDENCE and prob_gap >= UNCERTAIN_THRESHOLD:
        predicted_class = top1_class
        confidence = top1_prob
        is_certain = True
        risk_level = "High" if confidence >= HIGH_RISK_THRESHOLD else "Medium" if confidence >= MEDIUM_RISK_THRESHOLD else "Low"
    elif top1_prob < MIN_CONFIDENCE and prob_gap >= UNCERTAIN_THRESHOLD:
        predicted_class = "Uncertain"
        confidence = top1_prob
        is_certain = False
        risk_level = "Requires Review"
    elif prob_gap < UNCERTAIN_THRESHOLD:
        if top1_class == "COVID" or top2_class == "COVID":
            predicted_class = "Suspicious - Needs Medical Review"
            risk_level = "High"
        else:
            predicted_class = "Uncertain - Further Analysis Required"
            risk_level = "Medium"
        confidence = top1_prob
        is_certain = False
    elif top1_prob < 50:
        predicted_class = "Low Quality / Insufficient Data"
        confidence = top1_prob
        is_certain = False
        risk_level = "Requires Retake"
    else:
        predicted_class = f"{top1_class} (Low Confidence)"
        confidence = top1_prob
        is_certain = False
        risk_level = "Low"
    
    return predicted_class, confidence, risk_level, is_certain, probabilities, class_idx

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'alive',
        'model_loaded': model_loaded,
        'loading': loading,
        'heatmap_enabled': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, model_loaded
    
    if not model_loaded:
        if loading:
            return jsonify({'success': False, 'error': 'Model is loading'}), 503
        else:
            threading.Thread(target=load_model).start()
            return jsonify({'success': False, 'error': 'Model loading started'}), 202
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    temp_path = os.path.join(TEMP_DIR, f"{int(time.time())}_{secure_filename(file.filename)}")
    file.save(temp_path)
    
    heatmap_url = None
    
    try:
        processed_img = preprocess_image(temp_path)
        if processed_img is None:
            return jsonify({'success': False, 'error': 'Failed to preprocess image'}), 400
        
        predictions = model.predict(processed_img, verbose=0)[0]
        predicted_class, confidence, risk_level, is_certain, probabilities, class_idx = make_smart_prediction(predictions)
        
        # Generate heatmap
        print(f"🔥 Generating heatmap for class {class_idx} ({class_names[class_idx]})")
        heatmap = generate_gradcam(processed_img, class_idx)
        if heatmap is not None:
            heatmap_url = overlay_heatmap(temp_path, heatmap, predicted_class, confidence)
            if heatmap_url:
                print(f"✅ Heatmap saved: {heatmap_url}")
        
        recommendations = generate_recommendations(predicted_class, confidence, is_certain)
        
        if is_certain:
            message = f"Detected: {predicted_class} with {confidence:.1f}% confidence"
        else:
            message = f"⚠️ {predicted_class} (Confidence: {confidence:.1f}%)"
        
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'probabilities': probabilities,
            'recommendations': recommendations,
            'is_certain': is_certain,
            'message': message
        }
        
        if heatmap_url:
            response['heatmap'] = heatmap_url
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

def generate_recommendations(prediction, confidence, is_certain):
    if not is_certain:
        return [
            "⚠️ The AI model is uncertain about this X-ray image",
            "Please ensure the X-ray is of good quality",
            "🚨 Consult a healthcare provider for confirmation"
        ]
    
    if "Normal" in prediction:
        return ["✅ Your X-ray appears normal", "Continue maintaining a healthy lifestyle", "Annual check-ups advised"]
    elif "Pneumonia" in prediction:
        recs = ["⚠️ Consult a healthcare provider immediately", "Get plenty of rest and stay hydrated"]
        if confidence > 85:
            recs.insert(0, "🚨 HIGH CONFIDENCE: Immediate medical attention")
        return recs
    elif "COVID" in prediction or "Suspicious" in prediction:
        return ["🚨 URGENT: Isolate immediately", "Get tested for COVID-19", "Monitor oxygen levels"]
    else:
        return ["Consult with a healthcare provider", "Share this AI analysis with your doctor"]

@app.route('/static/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    threading.Thread(target=load_model).start()
    print("=" * 60)
    print("🚀 LUNG X-RAY MODEL SERVER (Grad-CAM Enhanced)")
    print("=" * 60)
    print(f"📁 Heatmaps: {STATIC_DIR}")
    print(f"⚙️  Min confidence: {MIN_CONFIDENCE}%")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)