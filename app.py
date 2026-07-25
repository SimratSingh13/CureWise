from flask import Flask, render_template, session, redirect, url_for, request, jsonify, flash, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore
from functools import wraps
import os
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import traceback
import threading
import requests
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from modules.predict_diabetes import DiabetesPredictor, predict_diabetes
from modules.ai_recommendations import AIRecommendationEngine

app = Flask(__name__)
app.secret_key = 'curewise-secret-key-2025'

cred = credentials.Certificate("curewise-firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

diabetes_predictor = DiabetesPredictor()
ai_engine = AIRecommendationEngine()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/heatmaps', exist_ok=True)
os.makedirs('temp', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

class ModelServiceClient:
    def __init__(self, server_url='http://127.0.0.1:5001'):
        self.server_url = server_url
        self.session = requests.Session()
    
    def check_health(self):
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=2)
            return response.json()
        except requests.exceptions.ConnectionError:
            return {'status': 'unreachable', 'model_loaded': False, 'loading': False}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'model_loaded': False, 'loading': False}
    
    def predict(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/png')}
                response = self.session.post(f"{self.server_url}/predict", files=files, timeout=120)
                return response.json()
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Model server not running. Please start it with: python model_server.py'}
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Model server timeout. The model might still be loading.'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_status(self):
        health = self.check_health()
        return {
            'loaded': health.get('model_loaded', False),
            'loading': health.get('loading', False),
            'server_status': health.get('status', 'unknown'),
            'model_path_exists': os.path.exists('models/lung_xray_model.h5')
        }

model_service = ModelServiceClient()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/auth/login', methods=['POST'])
def auth_login():
    try:
        data = request.json
        session.clear()
        session['user_id'] = data['uid']
        session['user_email'] = data['email']
        session['user_name'] = data.get('name', '')
        session['user_photo'] = data.get('photo', '')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/auth/signup', methods=['POST'])
def auth_signup():
    try:
        data = request.json
        user_ref = db.collection('users').document(data['uid'])
        user_ref.set({
            'uid': data['uid'],
            'email': data['email'],
            'name': data.get('name', ''),
            'photo': data.get('photo', ''),
            'created_at': firestore.SERVER_TIMESTAMP
        })
        session.clear()
        session['user_id'] = data['uid']
        session['user_email'] = data['email']
        session['user_name'] = data.get('name', '')
        session['user_photo'] = data.get('photo', '')
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', 
                         user_id=session.get('user_id'),
                         user_email=session.get('user_email'),
                         user_name=session.get('user_name'),
                         user_photo=session.get('user_photo'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    try:
        user_ref = db.collection('users').document(session['user_id'])
        user_data = user_ref.get().to_dict() or {}
        
        if request.method == 'POST':
            name = request.form.get('name')
            dob = request.form.get('dob')
            gender = request.form.get('gender')
            blood_group = request.form.get('blood_group')
            height = request.form.get('height')
            weight = request.form.get('weight')
            bp_systolic = request.form.get('bp_systolic')
            bp_diastolic = request.form.get('bp_diastolic')
            blood_sugar = request.form.get('blood_sugar')
            cholesterol = request.form.get('cholesterol')
            conditions = request.form.getlist('conditions')
            smoking = request.form.get('smoking')
            alcohol = request.form.get('alcohol')
            exercise = request.form.get('exercise')
            sleep = request.form.get('sleep')
            emergency_name = request.form.get('emergency_name')
            emergency_relation = request.form.get('emergency_relation')
            emergency_phone = request.form.get('emergency_phone')
            
            age = None
            if dob:
                birth_date = datetime.strptime(dob, '%Y-%m-%d')
                today = datetime.now()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            bmi = None
            if height and weight:
                try:
                    height_m = float(height) / 100
                    weight_kg = float(weight)
                    bmi = round(weight_kg / (height_m ** 2), 1)
                except:
                    bmi = None
            
            update_data = {
                'name': name if name else session.get('user_name'),
                'dob': dob,
                'gender': gender,
                'blood_group': blood_group,
                'age': age,
                'height': float(height) if height else None,
                'weight': float(weight) if weight else None,
                'bmi': bmi,
                'bp_systolic': int(bp_systolic) if bp_systolic else None,
                'bp_diastolic': int(bp_diastolic) if bp_diastolic else None,
                'blood_sugar': float(blood_sugar) if blood_sugar else None,
                'cholesterol': float(cholesterol) if cholesterol else None,
                'conditions': conditions,
                'smoking': smoking,
                'alcohol': alcohol,
                'exercise': exercise,
                'sleep': float(sleep) if sleep else None,
                'emergency_name': emergency_name,
                'emergency_relation': emergency_relation,
                'emergency_phone': emergency_phone,
                'profile_updated': firestore.SERVER_TIMESTAMP
            }
            
            update_data = {k: v for k, v in update_data.items() if v is not None}
            user_ref.update(update_data)
            
            if name and name != session.get('user_name'):
                session['user_name'] = name
            
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        
        age = None
        if user_data.get('dob'):
            try:
                birth_date = datetime.strptime(user_data['dob'], '%Y-%m-%d')
                today = datetime.now()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except:
                age = None
        
        if not user_data.get('bmi') and user_data.get('height') and user_data.get('weight'):
            try:
                height_m = float(user_data['height']) / 100
                weight_kg = float(user_data['weight'])
                user_data['bmi'] = round(weight_kg / (height_m ** 2), 1)
            except:
                pass
        
                # Count predictions for this user
        preds = db.collection('predictions').where('user_id', '==', session['user_id']).get()
        predictions_count = len(list(preds))
        
        return render_template('profile.html', user=user_data, age=age, predictions_count=predictions_count)
    
    except Exception as e:
        print("Profile error:", str(e))
        return f"Error loading profile: {str(e)}", 500

@app.route('/delete-account')
@login_required
def delete_account():
    try:
        user_id = session['user_id']
        auth.delete_user(user_id)
        db.collection('users').document(user_id).delete()
        predictions = db.collection('predictions').where('user_id', '==', user_id).get()
        for pred in predictions:
            pred.reference.delete()
        session.clear()
        flash('Your account has been deleted successfully.', 'success')
        return redirect(url_for('home'))
    except Exception as e:
        print(f"Delete account error: {str(e)}")
        flash(f'Error deleting account: {str(e)}', 'danger')
        return redirect(url_for('profile'))

@app.route('/health-report', methods=['GET', 'POST'])
@login_required
def health_report():
    try:
        user_ref = db.collection('users').document(session['user_id'])
        user_data = user_ref.get().to_dict() or {}
        
        age = None
        if user_data.get('dob'):
            try:
                birth_date = datetime.strptime(user_data['dob'], '%Y-%m-%d')
                today = datetime.now()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                user_data['age'] = age
            except:
                pass
        
        if request.method == 'POST':
            pregnancies = int(request.form.get('pregnancies', 0))
            glucose = float(request.form.get('glucose', 100))
            bp_systolic = int(request.form.get('bp_systolic', 120))
            bp_diastolic = int(request.form.get('bp_diastolic', 80))
            skin_thickness = float(request.form.get('skin_thickness', 20))
            insulin = float(request.form.get('insulin', 80))
            bmi = float(request.form.get('bmi', 25))
            dpf = float(request.form.get('dpf', 0.5))
            age_input = int(request.form.get('age', 30))
            cholesterol = request.form.get('cholesterol')
            if cholesterol:
                cholesterol = float(cholesterol)
            weight = float(request.form.get('weight', 70))
            symptoms = request.form.get('symptoms', '')
            
            diabetes_result = diabetes_predictor.predict(
                pregnancies, glucose, bp_systolic, 
                skin_thickness, insulin, bmi, dpf, age_input
            )
            
            diabetes_confidence = diabetes_result.get('confidence', 50)
            
            if diabetes_confidence <= 35:
                diabetes_risk_level = "Low"
            elif diabetes_confidence <= 65:
                diabetes_risk_level = "Medium"
            else:
                diabetes_risk_level = "High"
            
            diabetes_result['risk_level'] = diabetes_risk_level
            diabetes_result['confidence'] = diabetes_confidence
            
            bmi_category = "Normal"
            if bmi >= 30:
                bmi_category = "Obese"
            elif bmi >= 25:
                bmi_category = "Overweight"
            elif bmi < 18.5:
                bmi_category = "Underweight"
            
            hypertension_confidence = 30
            hypertension_reasons = ["Blood pressure is within normal range"]
            
            if bp_systolic >= 140 or bp_diastolic >= 90:
                hypertension_confidence = 85
                hypertension_reasons = ["Blood pressure is in hypertensive range (>=140/90)"]
            elif bp_systolic >= 130 or bp_diastolic >= 80:
                hypertension_confidence = 60
                hypertension_reasons = ["Blood pressure is elevated (130-139/80-89)"]
            
            if hypertension_confidence <= 35:
                hypertension_risk = "Low"
            elif hypertension_confidence <= 65:
                hypertension_risk = "Medium"
            else:
                hypertension_risk = "High"
            
            risk_factors = 0
            heart_disease_reasons = []
            
            if bmi > 30:
                risk_factors += 1
                heart_disease_reasons.append("High BMI (>30)")
            if cholesterol and cholesterol > 240:
                risk_factors += 1
                heart_disease_reasons.append("High cholesterol (>240 mg/dL)")
            if bp_systolic > 140:
                risk_factors += 1
                heart_disease_reasons.append("High blood pressure")
            if age_input > 50:
                risk_factors += 1
                heart_disease_reasons.append("Age over 50")
            if diabetes_result.get('prediction') == 1:
                risk_factors += 2
                heart_disease_reasons.append("Diabetes increases heart disease risk")
            
            if risk_factors >= 4:
                heart_confidence = 85
            elif risk_factors >= 2:
                heart_confidence = 55
            else:
                heart_confidence = 25
                if not heart_disease_reasons:
                    heart_disease_reasons = ["No major risk factors detected"]
            
            if heart_confidence <= 35:
                heart_disease_risk = "Low"
            elif heart_confidence <= 65:
                heart_disease_risk = "Medium"
            else:
                heart_disease_risk = "High"
            
            overall_score = 75
            if diabetes_result.get('prediction') == 1:
                overall_score -= 20
            if hypertension_risk == "High":
                overall_score -= 15
            elif hypertension_risk == "Medium":
                overall_score -= 8
            if heart_disease_risk == "High":
                overall_score -= 15
            elif heart_disease_risk == "Medium":
                overall_score -= 8
            if bmi > 30:
                overall_score -= 10
            elif bmi > 25:
                overall_score -= 5
            
            overall_score = max(0, min(100, overall_score))
            
            if overall_score >= 80:
                overall_status = "Excellent"
            elif overall_score >= 60:
                overall_status = "Good"
            elif overall_score >= 40:
                overall_status = "Fair"
            else:
                overall_status = "Needs Attention"
            
            recs = diabetes_result.get('recommendations', [])
            if not isinstance(recs, list):
                if isinstance(recs, (int, float)):
                    recs = [str(recs)]
                else:
                    recs = []
            
            if not recs:
                recs = [
                    'Monitor blood sugar levels regularly',
                    'Maintain a healthy diet low in sugar',
                    'Exercise for at least 30 minutes daily',
                    'Consult with a healthcare provider',
                    'Regular check-ups are recommended'
                ]
            
            result = {
                'bmi': bmi,
                'bmi_category': bmi_category,
                'cholesterol': cholesterol,
                'symptoms': symptoms,
                'diabetes_input': {
                    'pregnancies': pregnancies,
                    'glucose': glucose,
                    'blood_pressure': bp_systolic,
                    'bp_diastolic': bp_diastolic,
                    'skin_thickness': skin_thickness,
                    'insulin': insulin,
                    'bmi': bmi,
                    'dpf': dpf,
                    'age': age_input
                },
                'diabetes': {
                    'prediction': int(diabetes_result.get('prediction', 0)),
                    'risk_level': diabetes_result.get('risk_level', 'Low'),
                    'confidence': diabetes_confidence,
                    'message': diabetes_result.get('message', 'No message'),
                    'reasons': recs[:3],
                    'recommendations': recs
                },
                'hypertension': {
                    'risk_level': hypertension_risk,
                    'confidence': hypertension_confidence,
                    'reasons': hypertension_reasons
                },
                'heart_disease': {
                    'risk_level': heart_disease_risk,
                    'confidence': heart_confidence,
                    'reasons': heart_disease_reasons
                },
                'overall_score': overall_score,
                'overall_status': overall_status,
                'recommendations': [
                    'Maintain a balanced diet rich in fruits and vegetables',
                    'Exercise for at least 30 minutes daily',
                    'Stay hydrated and limit processed foods',
                    'Get at least 7-8 hours of sleep',
                    'Manage stress through meditation or yoga',
                    'Regular health check-ups are essential'
                ]
            }
            
            if diabetes_result.get('risk_level') == 'High':
                result['recommendations'].insert(0, 'URGENT: Consult a doctor immediately for diabetes management')
                result['recommendations'].insert(1, 'Monitor blood glucose levels daily')
            
            try:
                prediction_ref = db.collection('predictions').document()
                prediction_ref.set({
                    'user_id': session['user_id'],
                    'type': 'complete_health_report',
                    'date': firestore.SERVER_TIMESTAMP,
                    'result': result
                })
            except Exception as e:
                print(f"Error saving: {str(e)}")
            
            return render_template('report_result.html', 
                                 user=user_data, 
                                 result=result,
                                 request=request,
                                 datetime=datetime)
        
        return render_template('report_form.html', user=user_data, age=age)
    
    except Exception as e:
        print("Health report error:", str(e))
        traceback.print_exc()
        flash(f'Error: {str(e)}', 'danger')
        return render_template('report_form.html', user={}, age=None)

@app.route('/diabetes', methods=['GET', 'POST'])
@login_required
def diabetes_prediction():
    try:
        user_ref = db.collection('users').document(session['user_id'])
        user_data = user_ref.get().to_dict() or {}
        
        if user_data.get('dob'):
            try:
                birth_date = datetime.strptime(user_data['dob'], '%Y-%m-%d')
                today = datetime.now()
                user_data['age'] = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except:
                pass
        
        if request.method == 'POST':
            pregnancies = request.form.get('pregnancies', type=int)
            glucose = request.form.get('glucose', type=float)
            blood_pressure = request.form.get('blood_pressure', type=float)
            skin_thickness = request.form.get('skin_thickness', type=float)
            insulin = request.form.get('insulin', type=float)
            bmi = request.form.get('bmi', type=float)
            dpf = request.form.get('dpf', type=float)
            age = request.form.get('age', type=int)
            
            if None in [pregnancies, glucose, blood_pressure, skin_thickness, 
                       insulin, bmi, dpf, age]:
                flash('Please fill in all fields', 'danger')
                return render_template('diabetes_form.html', user=user_data)
            
            result = diabetes_predictor.predict(
                pregnancies, glucose, blood_pressure, 
                skin_thickness, insulin, bmi, dpf, age
            )
            
            if result.get('error'):
                flash(result['error'], 'danger')
                return render_template('diabetes_form.html', user=user_data)
            
            try:
                prediction_ref = db.collection('predictions').document()
                prediction_ref.set({
                    'user_id': session['user_id'],
                    'type': 'diabetes',
                    'date': firestore.SERVER_TIMESTAMP,
                    'input_data': {
                        'pregnancies': pregnancies,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'skin_thickness': skin_thickness,
                        'insulin': insulin,
                        'bmi': bmi,
                        'dpf': dpf,
                        'age': age
                    },
                    'result': {
                        'prediction': result['prediction'],
                        'risk_level': result['risk_level'],
                        'message': result['message'],
                        'confidence': result.get('confidence', 'N/A')
                    }
                })
            except Exception as e:
                print(f"Error saving prediction: {str(e)}")
            
            return render_template('diabetes_result.html', result=result, user=user_data)
        
        return render_template('diabetes_form.html', user=user_data)
    
    except Exception as e:
        print(f"Diabetes prediction error: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')
        return render_template('diabetes_form.html', user={})

@app.route('/api/predict-diabetes', methods=['POST'])
@login_required
def api_predict_diabetes():
    try:
        data = request.json
        pregnancies = data.get('pregnancies')
        glucose = data.get('glucose')
        blood_pressure = data.get('blood_pressure')
        skin_thickness = data.get('skin_thickness')
        insulin = data.get('insulin')
        bmi = data.get('bmi')
        dpf = data.get('dpf')
        age = data.get('age')
        
        if None in [pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, dpf, age]:
            return jsonify({'success': False, 'error': 'Missing fields'}), 400
        
        result = diabetes_predictor.predict(
            pregnancies, glucose, blood_pressure, 
            skin_thickness, insulin, bmi, dpf, age
        )
        
        if result.get('error'):
            return jsonify({'success': False, 'error': result['error']}), 500
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history')
@login_required
def history():
    try:
        user_id = session['user_id']
        predictions_ref = db.collection('predictions').where('user_id', '==', user_id).limit(200)
        predictions = list(predictions_ref.stream())
        predictions.sort(key=lambda x: x.to_dict().get('date', datetime.min), reverse=True)
        
        prediction_list = []
        xray_count = 0
        health_count = 0
        
        for pred in predictions:
            data = pred.to_dict()
            data['id'] = pred.id
            
            if data.get('date'):
                if hasattr(data['date'], 'strftime'):
                    data['date_formatted'] = data['date'].strftime('%Y-%m-%d %H:%M')
                else:
                    data['date_formatted'] = str(data['date'])
            else:
                data['date_formatted'] = 'Unknown date'
            
            if data.get('type') == 'lung_xray':
                xray_count += 1
            elif data.get('type') == 'complete_health_report':
                health_count += 1
            
            prediction_list.append(data)
        
        return render_template('history.html', 
                              predictions=prediction_list,
                              xray_count=xray_count,
                              health_count=health_count)
    
    except Exception as e:
        print(f"History error: {str(e)}")
        traceback.print_exc()
        return render_template('history.html', predictions=[], xray_count=0, health_count=0)

@app.route('/lung-analysis', methods=['GET', 'POST'])
@login_required
def lung_analysis():
    if request.method == 'POST':
        try:
            if 'xray_image' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
            file = request.files['xray_image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG, JPEG, or PNG images only.'}), 400
            
            status = model_service.get_status()
            
            if status['server_status'] == 'unreachable':
                return jsonify({'success': False, 'error': 'Model server is not running. Please start it with: python model_server.py'}), 503
            
            if status['loading']:
                return jsonify({'success': False, 'error': 'Model is still loading. Please wait a few seconds and try again.'}), 503
            
            if not status['loaded']:
                return jsonify({'success': False, 'error': 'Model not loaded. Please check model server.'}), 503
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = secure_filename(file.filename)
            filename = f"{session['user_id']}_{timestamp}_{safe_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            result = model_service.predict(filepath)
            
            # --- NEW CODE: Convert heatmap to base64 if it exists ---
            heatmap_base64 = None
            if result.get('success') and result.get('heatmap'):
                heatmap_url = result.get('heatmap')
                # Extract filename from URL: "/static/heatmaps/heatmap_xxx.jpg" -> "heatmap_xxx.jpg"
                heatmap_filename = heatmap_url.split('/')[-1]
                heatmap_path = os.path.join('static', 'heatmaps', heatmap_filename)
                
                if os.path.exists(heatmap_path):
                    with open(heatmap_path, "rb") as heatmap_file:
                        heatmap_base64 = base64.b64encode(heatmap_file.read()).decode('utf-8')
                    print(f"✅ Heatmap converted to base64: {heatmap_filename}")
                else:
                    print(f"⚠️ Heatmap file not found: {heatmap_path}")
            # --- END NEW CODE ---
            
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            if not result.get('success'):
                return jsonify({'success': False, 'error': result.get('error', 'Prediction failed')}), 500
            
            # --- UPDATED: Save heatmap_base64 to Firebase ---
            prediction_ref = db.collection('predictions').document()
            prediction_data = {
                'user_id': session['user_id'],
                'type': 'lung_xray',
                'date': firestore.SERVER_TIMESTAMP,
                'image_filename': filename,
                'image_base64': image_base64,
                'heatmap_base64': heatmap_base64,  # ← NEW: Save base64 instead of URL
                'result': {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level'],
                    'probabilities': result.get('probabilities', {}),
                    'message': result['message'],
                    'is_certain': result.get('is_certain', False),
                    'recommendations': result.get('recommendations', []),
                    'explanation': result.get('explanation', '')
                }
            }
            prediction_ref.set(prediction_data)
            # --- END UPDATE ---
            
            # Also return heatmap_base64 in the response for immediate display
            response_result = result.copy()
            response_result['heatmap_base64'] = heatmap_base64
            
            return jsonify({'success': True, 'result': response_result})
            
        except Exception as e:
            print(f"Lung analysis error: {str(e)}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('lung_analysis.html')

@app.route('/model-status')
@login_required
def model_status():
    if diabetes_predictor.model is not None:
        return jsonify({
            'status': 'active',
            'accuracy': diabetes_predictor.accuracy if hasattr(diabetes_predictor, 'accuracy') else None,
            'message': 'Diabetes prediction model is loaded and ready'
        })
    else:
        return jsonify({
            'status': 'inactive',
            'message': 'Diabetes prediction model is not loaded. Please train the model first.'
        }), 503

@app.route('/lung-status')
def lung_status():
    try:
        status = model_service.get_status()
        status['confidence_thresholds'] = {
            'min_confidence': 60.0,
            'uncertain_gap': 15.0,
            'high_risk': 85.0,
            'medium_risk': 60.0
        }
        status['heatmap_enabled'] = True
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-model-server')
def check_model_server():
    try:
        status = model_service.get_status()
        return jsonify({
            'running': status['server_status'] != 'unreachable',
            'status': status
        })
    except Exception as e:
        return jsonify({'running': False, 'error': str(e)})

@app.route('/static/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory('static/heatmaps', filename)

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/department')
def department():
    return render_template('department.html')

@app.route('/department-single')
def department_single():
    return render_template('department-single.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/doctor-single')
def doctor_single():
    return render_template('doctor-single.html')

@app.route('/appointment')
def appointment():
    return render_template('appoinment.html')

@app.route('/blog')
def blog():
    return render_template('blog-sidebar.html')

@app.route('/blog-single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/view-report/<report_id>')
@login_required
def view_report(report_id):
    try:
        doc_ref = db.collection('predictions').document(report_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            flash('Report not found', 'danger')
            return redirect(url_for('history'))
        
        data = doc.to_dict()
        
        if data.get('user_id') != session['user_id']:
            flash('Unauthorized access', 'danger')
            return redirect(url_for('history'))
        
        if data.get('type') == 'complete_health_report':
            result = data.get('result', {})
            user_data = db.collection('users').document(session['user_id']).get().to_dict() or {}
            return render_template('report_result.html', 
                                 user=user_data, 
                                 result=result, 
                                 request=request,
                                 datetime=datetime)
        elif data.get('type') == 'lung_xray':
            return render_template('xray_result.html', result=data.get('result', {}))
        else:
            flash('Report type not supported', 'warning')
            return redirect(url_for('history'))
            
    except Exception as e:
        print(f"View report error: {str(e)}")
        flash(f'Error loading report: {str(e)}', 'danger')
        return redirect(url_for('history'))

@app.route('/api/send-report-email', methods=['POST'])
@login_required
def send_report_email():
    try:
        data = request.json
        email = data.get('email')
        report_html = data.get('reportHtml')
        
        if not email or not report_html:
            return jsonify({'success': False, 'error': 'Missing email or report'})
        
        user_ref = db.collection('users').document(session['user_id'])
        user_data = user_ref.get().to_dict() or {}
        user_name = session.get('user_name', 'User')
        
        try:
            db.collection('shared_reports').add({
                'user_id': session['user_id'],
                'sent_to': email,
                'sent_at': firestore.SERVER_TIMESTAMP,
                'report_type': 'health_report'
            })
        except:
            pass
        
        return jsonify({'success': True, 'message': f'Report sent to {email}'})
    
    except Exception as e:
        print(f"Email error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-ai-recommendations', methods=['POST'])
@login_required
def get_ai_recommendations():
    try:
        data = request.json
        health_data = {
            'diabetes_risk': data.get('diabetes_risk', 'Low'),
            'diabetes_confidence': data.get('diabetes_confidence', 50),
            'hypertension_risk': data.get('hypertension_risk', 'Low'),
            'heart_risk': data.get('heart_risk', 'Low'),
            'bmi': data.get('bmi', 25),
            'bmi_category': data.get('bmi_category', 'Normal'),
            'age': data.get('age', 30),
            'glucose': data.get('glucose', 100),
            'bp_systolic': data.get('bp_systolic', 120),
            'bp_diastolic': data.get('bp_diastolic', 80)
        }
        
        recommendations = ai_engine.generate_all_recommendations(health_data)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"AI recommendations error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CUREWISE HEALTH PLATFORM (Enhanced with Grad-CAM Heatmaps)")
    print("="*60)
    
    if diabetes_predictor.model is None:
        print("Warning: Diabetes model not loaded")
    else:
        print(f"Diabetes model loaded (accuracy: {diabetes_predictor.accuracy:.2%})")
    
    print("\n" + "="*50)
    print("LUNG X-RAY MODEL (MobileNetV2 + Grad-CAM Explainability)")
    print("="*50)
    print("Model file: lung_xray_model.h5")
    print("Overall Accuracy: 91.0%")
    print("  - Normal: 100.0%")
    print("  - Pneumonia: 73.0%")
    print("  - COVID: 70.0%")
    print("\nConfidence Configuration:")
    print("  - Minimum confidence: 60%")
    print("  - Uncertainty gap: 15%")
    print("  - High risk: >85%, Medium: 60-85%, Low: <60%")
    print("\nGrad-CAM Explainability:")
    print("  - Heatmaps generated for every prediction")
    print("  - Shows important regions influencing the AI decision")
    print("="*50)
    
    try:
        status = model_service.get_status()
        if status['server_status'] == 'unreachable':
            print("\nWarning: Model server is not running!")
            print("   Please start it in a separate terminal with:")
            print("   python model_server.py")
        elif status['loaded']:
            print("\nModel server running (model loaded with Grad-CAM support)")
        elif status['loading']:
            print("\nModel server running (model loading...)")
        else:
            print("\nModel server running but model not loaded")
    except:
        print("\nCould not check model server status")
    
    print("\nOpen http://127.0.0.1:5000")
    print("Lung Analysis: http://127.0.0.1:5000/lung-analysis")
    print("History page: http://127.0.0.1:5000/history")
    print("Check lung status: http://127.0.0.1:5000/lung-status")
    print("Check model server: http://127.0.0.1:5000/check-model-server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)