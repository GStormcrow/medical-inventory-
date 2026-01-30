# views.py - Single capture facial recognition (no streaming)
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Sum, Count
from django.db.models.functions import TruncDate
from django.core.files.storage import default_storage
import face_recognition
import pickle
import requests
import json
from datetime import timedelta
from .models import Astronaut, Medication, Prescription, MedicationCheckout, InventoryLog, SystemLog
import serial, time

# Configuration
ESP32_PORT = "COM5"
BAUD_RATE = 115200

try:
    esp32_serial = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # Wait for connection to establish
    print("Connected to ESP32 on port", ESP32_PORT)
except serial.SerialException as e:
    print(f"Error connecting to ESP32: {e}")
    esp32_serial = None
    
def send_unlock_signal():
    """Send unlock signal to ESP32 via serial"""
    if esp32_serial:
        try:
            esp32_serial.write(b'unlock')  # unlock command
            response = esp32_serial.readline().decode().strip()
            return response == 'UNLOCKED'
        except Exception as e:
            print(f"Error sending unlock signal: {e}")
            return False
    return False 

# def send_lock_signal():
#     """Send lock signal to ESP32 via serial"""
#     if esp32_serial:
#         try:
#             esp32_serial.write(b'lock')  # unlock command
#             response = esp32_serial.readline().decode().strip()
#             return response == 'LOCKED'
#         except Exception as e:
#             print(f"Error sending unlock signal: {e}")
#             return False
#     return False

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
                        
                        unlocked_sent = send_unlock_signal()
                        
                        SystemLog.objects.create(
                            event_type='AUTH_SUCCESS',
                            astronaut=astronaut,
                            description=f"Astronaut {astronaut.name} successfully authenticated",
                            ip_address=request.META.get('REMOTE_ADDR')
                        )
                        
                        return JsonResponse({
                            'success': True,
                            'astronaut_id': astronaut.id,
                            'astronaut_name': astronaut.name,
                            'door_unlocked': unlocked_sent,
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
    
    inventory_logs = InventoryLog.objects.filter(medication=medication)[:20]
    
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
    """API endpoint for pill recognition using uploaded image"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Save uploaded image temporarily
            image_path = default_storage.save(f'temp/{image_file.name}', image_file)
            full_path = default_storage.path(image_path)
            
            # TODO: Load your trained pill recognition model here
            # For now, return placeholder response
            
            # Clean up temp file
            default_storage.delete(image_path)
            
            return JsonResponse({
                'success': False,
                'message': 'Pill recognition model not yet implemented. Upload your trained model to enable this feature.'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error processing image: {str(e)}'
            })
    
    return JsonResponse({'error': 'No image provided'}, status=400)


# ===== ADMIN API ENDPOINTS =====

@csrf_exempt
def add_astronaut(request):
    """Add new astronaut with face photo"""
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            astronaut_id = request.POST.get('astronaut_id')
            password = request.POST.get('password', '')
            photo = request.FILES.get('photo')
            
            if not all([name, astronaut_id, photo]):
                return JsonResponse({
                    'success': False,
                    'message': 'Name, Astronaut ID, and photo are required'
                })
            
            # Create user account
            from django.contrib.auth.models import User
            username = astronaut_id.lower()
            user = User.objects.create_user(
                username=username,
                email=f"{username}@nasa.gov",  # Auto-generate email
                password=password if password else astronaut_id
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


def manage_astronauts(request):
    """Astronaut management page"""
    return render(request, 'manage_astronauts.html')


def manage_medications(request):
    """Medication management page"""
    return render(request, 'manage_medications.html')