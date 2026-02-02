# views.py - Complete Medical Inventory System with Facial Recognition and Pill Recognition
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Sum, Count
from django.db.models.functions import TruncDate
from django.core.files.storage import default_storage
from PIL import Image
import face_recognition
import pickle
import requests
import json
import numpy as np
import cv2
import os
from datetime import timedelta
from .models import Astronaut, Medication, Prescription, MedicationCheckout, InventoryLog, SystemLog

# Import for deep learning model (TensorFlow/Keras)
try:
    from tensorflow import keras
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# Import for color/shape analysis
from sklearn.cluster import KMeans

# Configuration
ESP32_IP = "192.168.1.100"


# ============================================================================
# HOME AND AUTHENTICATION VIEWS
# ============================================================================

def home(request):
    """Home screen"""
    return render(request, 'home.html')


def lockscreen(request):
    """Lockscreen with single-capture facial recognition"""
    return render(request, 'lockscreen.html')


@csrf_exempt
def authenticate_face(request):
    """Single image capture face authentication"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Load image with face_recognition
            image = face_recognition.load_image_file(image_file)
            
            # Find faces in the uploaded image
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                SystemLog.objects.create(
                    event_type='AUTH_FAILURE',
                    description="No face detected in image",
                    ip_address=request.META.get('REMOTE_ADDR')
                )
                return JsonResponse({
                    'success': False,
                    'message': 'No face detected. Please ensure your face is clearly visible.'
                })
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                return JsonResponse({
                    'success': False,
                    'message': 'Could not process face. Please try again.'
                })
            
            # Load known faces from database
            astronauts = Astronaut.objects.exclude(face_encoding__isnull=True)
            
            for face_encoding in face_encodings:
                for astronaut in astronauts:
                    known_encoding = pickle.loads(astronaut.face_encoding)
                    
                    # Compare faces
                    matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                    
                    if matches[0]:
                        # Face recognized!
                        SystemLog.objects.create(
                            event_type='AUTH_SUCCESS',
                            astronaut=astronaut,
                            description=f"Astronaut {astronaut.name} successfully authenticated",
                            ip_address=request.META.get('REMOTE_ADDR')
                        )
                        
                        return JsonResponse({
                            'success': True,
                            'astronaut_id': astronaut.id,
                            'astronaut_name': astronaut.name
                        })
            
            # No match found
            SystemLog.objects.create(
                event_type='AUTH_FAILURE',
                description="Face not recognized - unknown individual",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            return JsonResponse({
                'success': False,
                'message': 'Face not recognized. Please ensure you are an authorized user.'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Authentication error: {str(e)}'
            })
    
    return JsonResponse({'error': 'Invalid request - image required'}, status=400)


# ============================================================================
# MEDICATION CHECKOUT VIEWS
# ============================================================================

def medication_selection(request, astronaut_id):
    """Display medication selection page after authentication"""
    astronaut = get_object_or_404(Astronaut, id=astronaut_id)
    
    prescriptions = Prescription.objects.filter(
        astronaut=astronaut,
        is_active=True
    ).select_related('medication')
    
    all_medications = Medication.objects.filter(current_quantity__gt=0)
    
    context = {
        'astronaut': astronaut,
        'prescriptions': prescriptions,
        'all_medications': all_medications
    }
    
    return render(request, 'medication_selection.html', context)


@csrf_exempt
def checkout_medication(request):
    """Process medication checkout"""
    if request.method == 'POST':
        data = json.loads(request.body)
        astronaut_id = data.get('astronaut_id')
        medications = data.get('medications', [])
        
        astronaut = get_object_or_404(Astronaut, id=astronaut_id)
        checkouts = []
        
        for med_data in medications:
            medication = get_object_or_404(Medication, id=med_data['medication_id'])
            quantity = med_data.get('quantity', 1)
            
            checkout = MedicationCheckout.objects.create(
                astronaut=astronaut,
                medication=medication,
                quantity=quantity,
                is_prescription=med_data.get('is_prescription', False)
            )
            checkouts.append(checkout)
            
            InventoryLog.objects.create(
                medication=medication,
                log_type='CHECKOUT',
                quantity_change=-quantity,
                previous_quantity=medication.current_quantity + quantity,
                new_quantity=medication.current_quantity,
                performed_by=astronaut,
                notes=f"Checkout by {astronaut.name}"
            )
        
        unlock_success = unlock_container(astronaut)
        
        SystemLog.objects.create(
            event_type='CONTAINER_UNLOCK',
            astronaut=astronaut,
            description=f"Container unlocked for {astronaut.name}. Status: {'Success' if unlock_success else 'Failed'}",
            ip_address=request.META.get('REMOTE_ADDR')
        )
        
        return JsonResponse({
            'success': True,
            'checkouts': len(checkouts),
            'unlock_status': unlock_success
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def unlock_container(astronaut):
    """Send unlock signal to ESP32"""
    try:
        url = f"http://{ESP32_IP}/unlock"
        payload = {
            'astronaut_id': astronaut.astronaut_id,
            'timestamp': timezone.now().isoformat()
        }
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error unlocking container: {e}")
        return False


# ============================================================================
# INVENTORY VIEWS
# ============================================================================

def inventory_dashboard(request):
    """Display full inventory with statistics"""
    medications = Medication.objects.all().order_by('name')
    
    total_medications = medications.count()
    low_stock_count = sum(1 for med in medications if med.is_low_stock)
    total_checkouts_today = MedicationCheckout.objects.filter(
        checkout_time__date=timezone.now().date()
    ).count()
    
    context = {
        'medications': medications,
        'total_medications': total_medications,
        'low_stock_count': low_stock_count,
        'total_checkouts_today': total_checkouts_today
    }
    
    return render(request, 'inventory_dashboard.html', context)


def medication_detail(request, medication_id):
    """Detailed view of a specific medication with usage statistics"""
    medication = get_object_or_404(Medication, id=medication_id)
    
    thirty_days_ago = timezone.now() - timedelta(days=30)
    checkouts = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).order_by('-checkout_time')
    
    daily_usage = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).annotate(
        date=TruncDate('checkout_time')
    ).values('date').annotate(
        total_quantity=Sum('quantity'),
        checkout_count=Count('id')
    ).order_by('date')
    
    inventory_logs = InventoryLog.objects.filter(
        medication=medication
    ).order_by('-timestamp')[:10]
    
    context = {
        'medication': medication,
        'checkouts': checkouts,
        'daily_usage': daily_usage,
        'inventory_logs': inventory_logs
    }
    
    return render(request, 'medication_detail.html', context)


# ============================================================================
# ASTRONAUT MANAGEMENT VIEWS
# ============================================================================

def manage_astronauts(request):
    """Astronaut management page"""
    return render(request, 'manage_astronauts.html')


@csrf_exempt
def add_astronaut(request):
    """Add new astronaut with face encoding"""
    if request.method == 'POST':
        try:
            astronaut_id = request.POST.get('astronaut_id')
            name = request.POST.get('name')
            email = request.POST.get('email')
            photo = request.FILES.get('photo')
            
            if not all([astronaut_id, name, email, photo]):
                return JsonResponse({
                    'success': False,
                    'message': 'All fields are required'
                })
            
            # Create user account
            from django.contrib.auth.models import User
            user = User.objects.create_user(
                username=astronaut_id,
                email=email,
                first_name=name.split()[0] if name else '',
                last_name=' '.join(name.split()[1:]) if len(name.split()) > 1 else ''
            )
            
            # Create astronaut
            astronaut = Astronaut.objects.create(
                user=user,
                astronaut_id=astronaut_id,
                name=name
            )
            
            # Process face encoding
            image = face_recognition.load_image_file(photo)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                astronaut.face_encoding = pickle.dumps(face_encodings[0])
                astronaut.save()
                
                return JsonResponse({
                    'success': True,
                    'message': 'Astronaut added successfully',
                    'astronaut_id': astronaut.id
                })
            else:
                astronaut.delete()
                user.delete()
                return JsonResponse({
                    'success': False,
                    'message': 'No face detected in photo. Please use a clear, front-facing photo.'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def list_astronauts(request):
    """List all astronauts"""
    astronauts = Astronaut.objects.all()
    
    data = [{
        'id': a.id,
        'name': a.name,
        'astronaut_id': a.astronaut_id,
        'has_face_encoding': a.face_encoding is not None,
        'photo_url': None  # We don't store photos, just encodings
    } for a in astronauts]
    
    return JsonResponse({'astronauts': data})


@csrf_exempt
def update_astronaut_face(request):
    """Update astronaut face encoding"""
    if request.method == 'POST':
        try:
            astronaut_id = request.POST.get('astronaut_id')
            photo = request.FILES.get('photo')
            
            if not all([astronaut_id, photo]):
                return JsonResponse({
                    'success': False,
                    'message': 'Astronaut ID and photo are required'
                })
            
            astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            
            # Process face encoding
            image = face_recognition.load_image_file(photo)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                astronaut.face_encoding = pickle.dumps(face_encodings[0])
                astronaut.save()
                
                return JsonResponse({
                    'success': True,
                    'message': 'Face encoding updated successfully'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'No face detected in photo'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def delete_astronaut(request, astronaut_id):
    """Delete astronaut"""
    if request.method == 'DELETE':
        try:
            astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            user = astronaut.user
            astronaut.delete()
            user.delete()
            
            return JsonResponse({
                'success': True,
                'message': 'Astronaut deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# ============================================================================
# MEDICATION MANAGEMENT VIEWS
# ============================================================================

def manage_medications(request):
    """Medication management page"""
    return render(request, 'manage_medications.html')


def pill_recognition(request):
    """Pill recognition page for scanning and identifying pills"""
    return render(request, 'pill_recognition.html')


@csrf_exempt
def add_medication(request):
    """Add new medication with image"""
    if request.method == 'POST':
        try:
            medication = Medication.objects.create(
                name=request.POST.get('name'),
                generic_name=request.POST.get('generic_name', ''),
                medication_type=request.POST.get('medication_type'),
                dosage=request.POST.get('dosage'),
                description=request.POST.get('description', ''),
                current_quantity=int(request.POST.get('current_quantity', 0)),
                minimum_quantity=int(request.POST.get('minimum_quantity', 0)),
                container_location=request.POST.get('container_location'),
                expiration_date=request.POST.get('expiration_date') or None,
                pill_image=request.FILES.get('pill_image')
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Medication added successfully',
                'medication_id': medication.id
            })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt  
def list_medications(request):
    """List all medications"""
    medications = Medication.objects.all()
    
    data = [{
        'id': m.id,
        'name': m.name,
        'generic_name': m.generic_name,
        'medication_type': m.get_medication_type_display(),
        'dosage': m.dosage,
        'current_quantity': m.current_quantity,
        'minimum_quantity': m.minimum_quantity,
        'container_location': m.container_location,
        'expiration_date': m.expiration_date.strftime('%Y-%m-%d') if m.expiration_date else None,
        'image_url': m.pill_image.url if m.pill_image else None
    } for m in medications]
    
    return JsonResponse({'medications': data})


@csrf_exempt
def update_medication_image(request):
    """Update medication image"""
    if request.method == 'POST':
        try:
            medication_id = request.POST.get('medication_id')
            image = request.FILES.get('image')
            
            if not all([medication_id, image]):
                return JsonResponse({
                    'success': False,
                    'message': 'Medication ID and image are required'
                })
            
            medication = get_object_or_404(Medication, id=medication_id)
            medication.pill_image = image
            medication.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Image updated successfully',
                'image_url': medication.pill_image.url
            })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def delete_medication(request, medication_id):
    """Delete medication"""
    if request.method == 'DELETE':
        try:
            medication = get_object_or_404(Medication, id=medication_id)
            medication.delete()
            
            return JsonResponse({
                'success': True,
                'message': 'Medication deleted successfully'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


# ============================================================================
# PILL RECOGNITION - Deep Learning CNN Model (Most Accurate)
# ============================================================================

def load_pill_recognition_model():
    """Load the trained CNN model for pill recognition"""
    model_path = os.path.join('models', 'pill_recognition_model.h5')
    
    if os.path.exists(model_path) and TENSORFLOW_AVAILABLE:
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None


def preprocess_image_for_cnn(image_path, target_size=(224, 224)):
    """Preprocess image for CNN model"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def recognize_pill_with_cnn(image_path):
    """
    Recognize pill using trained CNN model
    Returns: (medication_id, confidence, medication_name)
    """
    model = load_pill_recognition_model()
    
    if model is None:
        return None, 0.0, "Model not loaded"
    
    # Preprocess image
    processed_image = preprocess_image_for_cnn(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class]) * 100
    
    # Load class labels (you need to create this file during training)
    labels_path = os.path.join('models', 'pill_labels.pkl')
    if os.path.exists(labels_path):
        with open(labels_path, 'rb') as f:
            class_labels = pickle.load(f)
        medication_name = class_labels[predicted_class]
    else:
        medication_name = f"Class_{predicted_class}"
    
    return predicted_class, confidence, medication_name


# ============================================================================
# PILL RECOGNITION - Color and Shape Analysis (Fallback Method)
# ============================================================================

def extract_pill_features(image_path):
    """
    Extract color and shape features from pill image
    Returns: dict with color, shape, and size information
    """
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for shape detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (assumed to be the pill)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Approximate shape
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    num_sides = len(approx)
    
    # Determine shape
    if num_sides < 6 and circularity > 0.7:
        shape = "ROUND"
    elif num_sides == 4:
        shape = "SQUARE"
    elif 4 < num_sides < 8:
        shape = "OVAL"
    else:
        shape = "CAPSULE"
    
    # Extract dominant colors
    img_flat = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(img_flat)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Convert RGB to color names
    color_name = rgb_to_color_name(dominant_colors[0])
    
    # Get bounding rectangle for size
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        'shape': shape,
        'color': color_name,
        'circularity': round(circularity, 2),
        'width': w,
        'height': h,
        'area': int(area),
        'num_sides': num_sides,
        'dominant_colors': dominant_colors.tolist()
    }


def rgb_to_color_name(rgb):
    """Convert RGB values to common color names"""
    r, g, b = rgb
    
    # Define color thresholds
    if r > 200 and g > 200 and b > 200:
        return "WHITE"
    elif r < 50 and g < 50 and b < 50:
        return "BLACK"
    elif r > 180 and g < 100 and b < 100:
        return "RED"
    elif r < 100 and g > 180 and b < 100:
        return "GREEN"
    elif r < 100 and g < 100 and b > 180:
        return "BLUE"
    elif r > 180 and g > 180 and b < 100:
        return "YELLOW"
    elif r > 180 and g > 100 and b < 100:
        return "ORANGE"
    elif r > 150 and g < 100 and b > 150:
        return "PINK"
    elif r > 120 and g > 60 and b < 60:
        return "BROWN"
    else:
        return "MULTI-COLOR"


def match_pill_by_features(features):
    """
    Match pill to database by color and shape features
    Returns: list of potential matches with confidence scores
    """
    if not features:
        return []
    
    # Query medications from database
    medications = Medication.objects.filter(current_quantity__gt=0)
    
    matches = []
    for med in medications:
        score = 0
        
        # Compare shape (stored in medication description or custom field)
        # You would need to add these fields to your Medication model
        if hasattr(med, 'pill_shape') and med.pill_shape == features['shape']:
            score += 40
        
        # Compare color
        if hasattr(med, 'pill_color') and med.pill_color == features['color']:
            score += 40
        
        # Compare size (if you have this data)
        if hasattr(med, 'pill_size'):
            # Add size comparison logic
            pass
        
        if score > 30:  # Threshold for potential match
            matches.append({
                'medication': med,
                'confidence': score,
                'reason': f"Matched on shape: {features['shape']}, color: {features['color']}"
            })
    
    # Sort by confidence
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches[:5]  # Return top 5 matches


# ============================================================================
# PILL RECOGNITION - Image Similarity Matching
# ============================================================================

def calculate_image_similarity(image1_path, image2_path):
    """
    Calculate similarity between two images using histogram comparison
    Returns: similarity score (0-100)
    """
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        return 0
    
    # Resize to same size
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    
    # Convert to HSV for better color comparison
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Compare histograms using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity * 100  # Convert to percentage


def match_pill_by_image(uploaded_image_path):
    """
    Match pill by comparing uploaded image to stored medication images
    Returns: list of matches with similarity scores
    """
    medications = Medication.objects.filter(
        current_quantity__gt=0,
        pill_image__isnull=False
    ).exclude(pill_image='')
    
    matches = []
    for med in medications:
        if med.pill_image:
            try:
                stored_image_path = med.pill_image.path
                similarity = calculate_image_similarity(uploaded_image_path, stored_image_path)
                
                if similarity > 60:  # Threshold for similarity
                    matches.append({
                        'medication': med,
                        'confidence': round(similarity, 2),
                        'reason': f"Visual similarity: {similarity:.1f}%"
                    })
            except Exception as e:
                print(f"Error comparing with {med.name}: {e}")
                continue
    
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches[:5]


# ============================================================================
# MAIN PILL RECOGNITION ENDPOINT
# ============================================================================

@csrf_exempt
def recognize_pill(request):
    """
    API endpoint for pill recognition using multiple approaches
    Tries: 1) CNN model, 2) Feature matching, 3) Image similarity
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Save uploaded image temporarily
            image_path = default_storage.save(f'temp/{image_file.name}', image_file)
            full_path = default_storage.path(image_path)
            
            results = {
                'success': False,
                'method_used': None,
                'matches': [],
                'features': None
            }
            
            # APPROACH 1: Try CNN model first (most accurate)
            if TENSORFLOW_AVAILABLE:
                try:
                    med_id, confidence, med_name = recognize_pill_with_cnn(full_path)
                    
                    if med_id is not None and confidence > 70:  # Confidence threshold
                        # Try to find medication in database
                        try:
                            medication = Medication.objects.get(id=med_id)
                            results['success'] = True
                            results['method_used'] = 'CNN_MODEL'
                            results['matches'] = [{
                                'medication_id': medication.id,
                                'name': medication.name,
                                'generic_name': medication.generic_name,
                                'dosage': medication.dosage,
                                'confidence': round(confidence, 2),
                                'current_quantity': medication.current_quantity,
                                'image_url': medication.pill_image.url if medication.pill_image else None
                            }]
                            
                            # Clean up and return
                            default_storage.delete(image_path)
                            return JsonResponse(results)
                        except Medication.DoesNotExist:
                            pass
                except Exception as e:
                    print(f"CNN recognition failed: {e}")
            
            # APPROACH 2: Feature-based matching
            try:
                features = extract_pill_features(full_path)
                
                if features:
                    results['features'] = features
                    feature_matches = match_pill_by_features(features)
                    
                    if feature_matches:
                        results['success'] = True
                        results['method_used'] = 'FEATURE_MATCHING'
                        results['matches'] = [{
                            'medication_id': match['medication'].id,
                            'name': match['medication'].name,
                            'generic_name': match['medication'].generic_name,
                            'dosage': match['medication'].dosage,
                            'confidence': match['confidence'],
                            'reason': match['reason'],
                            'current_quantity': match['medication'].current_quantity,
                            'image_url': match['medication'].pill_image.url if match['medication'].pill_image else None
                        } for match in feature_matches]
                        
                        # Clean up and return
                        default_storage.delete(image_path)
                        return JsonResponse(results)
            except Exception as e:
                print(f"Feature matching failed: {e}")
            
            # APPROACH 3: Image similarity matching
            try:
                similarity_matches = match_pill_by_image(full_path)
                
                if similarity_matches:
                    results['success'] = True
                    results['method_used'] = 'IMAGE_SIMILARITY'
                    results['matches'] = [{
                        'medication_id': match['medication'].id,
                        'name': match['medication'].name,
                        'generic_name': match['medication'].generic_name,
                        'dosage': match['medication'].dosage,
                        'confidence': match['confidence'],
                        'reason': match['reason'],
                        'current_quantity': match['medication'].current_quantity,
                        'image_url': match['medication'].pill_image.url if match['medication'].pill_image else None
                    } for match in similarity_matches]
                    
                    # Clean up and return
                    default_storage.delete(image_path)
                    return JsonResponse(results)
            except Exception as e:
                print(f"Image similarity matching failed: {e}")
            
            # No matches found
            default_storage.delete(image_path)
            return JsonResponse({
                'success': False,
                'message': 'No matching medication found. Please try again with a clearer image or select manually.',
                'features': results.get('features'),
                'method_used': 'NONE'
            })
            
        except Exception as e:
            # Clean up on error
            if 'image_path' in locals():
                try:
                    default_storage.delete(image_path)
                except:
                    pass
            
            return JsonResponse({
                'success': False,
                'message': f'Error processing image: {str(e)}'
            })
    
    return JsonResponse({'error': 'No image provided'}, status=400)