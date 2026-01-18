from django.shortcuts import render

# Create your views here.
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Sum, Count
from django.db.models.functions import TruncDate
import cv2
import face_recognition
import numpy as np
import pickle
import requests
import json
from datetime import timedelta
from .models import Astronaut, Medication, Prescription, MedicationCheckout, InventoryLog, SystemLog

# Configuration
ESP32_IP = "192.168.1.100"  # Update with your ESP32's IP
CAMERA_INDEX = 0


def home(request):
    """Home screen"""
    return render(request, 'home.html')


def lockscreen(request):
    """Lockscreen with facial recognition"""
    return render(request, 'lockscreen.html')


class VideoCamera:
    """Handle video streaming for facial recognition"""
    def __init__(self):
        self.video = cv2.VideoCapture(CAMERA_INDEX)
        self.known_face_encodings = []
        self.known_astronaut_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all astronaut face encodings from database"""
        astronauts = Astronaut.objects.exclude(face_encoding__isnull=True)
        for astronaut in astronauts:
            encoding = pickle.loads(astronaut.face_encoding)
            self.known_face_encodings.append(encoding)
            self.known_astronaut_ids.append(astronaut.id)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        """Capture frame and perform face recognition"""
        success, frame = self.video.read()
        if not success:
            return None, None
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_astronaut_id = None
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    recognized_astronaut_id = self.known_astronaut_ids[best_match_index]
                    
                    # Draw rectangle around face
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Authenticated", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    # Unknown face
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), recognized_astronaut_id


def gen_frames(camera):
    """Generator function for video streaming"""
    while True:
        frame, astronaut_id = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    """Video streaming route"""
    camera = VideoCamera()
    return StreamingHttpResponse(gen_frames(camera),
                                content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
def authenticate_face(request):
    """API endpoint to authenticate face and return astronaut data"""
    if request.method == 'POST':
        camera = VideoCamera()
        frame, astronaut_id = camera.get_frame()
        
        if astronaut_id:
            astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            
            # Log successful authentication
            SystemLog.objects.create(
                event_type='AUTH_SUCCESS',
                astronaut=astronaut,
                description=f"Astronaut {astronaut.name} successfully authenticated",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            # Get prescriptions and available medications
            prescriptions = Prescription.objects.filter(
                astronaut=astronaut,
                is_active=True
            ).select_related('medication')
            
            prescription_data = [{
                'id': p.medication.id,
                'name': p.medication.name,
                'dosage': p.prescribed_dosage,
                'frequency': p.frequency,
                'current_stock': p.medication.current_quantity
            } for p in prescriptions]
            
            return JsonResponse({
                'success': True,
                'astronaut_id': astronaut.id,
                'astronaut_name': astronaut.name,
                'prescriptions': prescription_data
            })
        else:
            # Log failed authentication
            SystemLog.objects.create(
                event_type='AUTH_FAILURE',
                description="Face recognition failed - unknown individual",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            return JsonResponse({
                'success': False,
                'message': 'Face not recognized'
            })
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


def medication_selection(request, astronaut_id):
    """Display medication selection page after authentication"""
    astronaut = get_object_or_404(Astronaut, id=astronaut_id)
    
    # Get prescribed medications
    prescriptions = Prescription.objects.filter(
        astronaut=astronaut,
        is_active=True
    ).select_related('medication')
    
    # Get all available medications for additional selection
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
        medications = data.get('medications', [])  # List of {medication_id, quantity, is_prescription}
        
        astronaut = get_object_or_404(Astronaut, id=astronaut_id)
        checkouts = []
        
        for med_data in medications:
            medication = get_object_or_404(Medication, id=med_data['medication_id'])
            quantity = med_data.get('quantity', 1)
            
            # Create checkout record
            checkout = MedicationCheckout.objects.create(
                astronaut=astronaut,
                medication=medication,
                quantity=quantity,
                is_prescription=med_data.get('is_prescription', False)
            )
            checkouts.append(checkout)
            
            # Create inventory log
            InventoryLog.objects.create(
                medication=medication,
                log_type='CHECKOUT',
                quantity_change=-quantity,
                previous_quantity=medication.current_quantity + quantity,
                new_quantity=medication.current_quantity,
                performed_by=astronaut,
                notes=f"Checkout by {astronaut.name}"
            )
        
        # Send unlock signal to ESP32
        unlock_success = unlock_container(astronaut)
        
        # Log container unlock
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


def inventory_dashboard(request):
    """Display full inventory with statistics"""
    medications = Medication.objects.all().order_by('name')
    
    # Calculate statistics
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
    
    # Get checkout history for last 30 days
    thirty_days_ago = timezone.now() - timedelta(days=30)
    checkouts = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).order_by('-checkout_time')
    
    # Aggregate daily usage for graphing
    daily_usage = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).annotate(
        date=TruncDate('checkout_time')
    ).values('date').annotate(
        total_quantity=Sum('quantity'),
        checkout_count=Count('id')
    ).order_by('date')
    
    # Get inventory logs
    inventory_logs = InventoryLog.objects.filter(
        medication=medication
    )[:20]
    
    context = {
        'medication': medication,
        'checkouts': checkouts,
        'daily_usage': list(daily_usage),
        'inventory_logs': inventory_logs,
        'total_dispensed_30d': sum(item['total_quantity'] for item in daily_usage)
    }
    
    return render(request, 'medication_detail.html', context)


def pill_recognition(request):
    """Page for pill/medication recognition via camera"""
    return render(request, 'pill_recognition.html')


@csrf_exempt
def recognize_pill(request):
    """API endpoint for pill recognition"""
    # This would use a trained model (TensorFlow/PyTorch)
    # For now, this is a placeholder
    if request.method == 'POST':
        # TODO: Implement pill recognition using computer vision
        # This would involve:
        # 1. Capture image from camera
        # 2. Preprocess image
        # 3. Run through trained model
        # 4. Return medication identification
        
        return JsonResponse({
            'success': False,
            'message': 'Pill recognition model not yet implemented'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)