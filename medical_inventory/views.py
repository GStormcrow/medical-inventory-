# views.py - Updated with Authentication, Camera Capture, Transaction Log
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.db.models import Sum, Count, Q
from django.db.models.functions import TruncDate
from django.core.files.storage import default_storage
from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from PIL import Image
import face_recognition
import pickle
import requests
import json
import numpy as np
import cv2
import csv
import hashlib
import io
import base64
import os
from datetime import timedelta
from .models import Astronaut, Medication, Prescription, MedicationCheckout, InventoryLog, SystemLog
from .forms import MedicationForm

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
# LOGIN/LOGOUT VIEWS
# ============================================================================

def login_view(request):
    """Login page for staff/admin users"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            next_url = request.GET.get('next', 'medical_inventory:home')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid credentials or insufficient permissions')
    
    return render(request, 'login.html')


def logout_view(request):
    """Logout view"""
    logout(request)
    return redirect('medical_inventory:home')


# ============================================================================
# HOME AND FACE AUTHENTICATION VIEWS
# ============================================================================

def home(request):
    """Home screen"""
    return render(request, 'home.html')


def lockscreen(request):
    """Lockscreen with facial recognition"""
    return render(request, 'lockscreen.html')


@csrf_exempt
def authenticate_face(request):
    """Face authentication endpoint"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            image = face_recognition.load_image_file(image_file)
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
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                return JsonResponse({
                    'success': False,
                    'message': 'Could not process face. Please try again.'
                })
            
            astronauts = Astronaut.objects.exclude(face_encoding__isnull=True)
            
            for face_encoding in face_encodings:
                for astronaut in astronauts:
                    known_encoding = pickle.loads(astronaut.face_encoding)
                    matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                    
                    if matches[0]:
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
# MEDICATION SELECTION AND CHECKOUT
# ============================================================================

def medication_selection(request, astronaut_id):
    """Medication selection page"""
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
    """Process medication checkout with transaction logging"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            astronaut_id = data.get('astronaut_id')
            medications = data.get('medications', [])
            
            astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            checkouts_created = 0
            # warnings_triggered = []
            
            for med_data in medications:
                medication = get_object_or_404(Medication, id=med_data['medication_id'])
                quantity = med_data['quantity']
                
                # Check quantity available
                if quantity > medication.current_quantity:
                    return JsonResponse({
                        'success': False,
                        'message': f'Insufficient stock for {medication.name}'
                    }, status=400)
                
                # Check thresholds
                # warning_created, severity = check_medication_threshold(astronaut, medication, quantity)
                # if warning_created:
                #     warnings_triggered.append({
                #         'medication': medication.name,
                #         'severity': severity
                #     })
                
                # Store previous quantity
                previous_quantity = medication.current_quantity
                
                # Create checkout
                MedicationCheckout.objects.create(
                    astronaut=astronaut,
                    medication=medication,
                    quantity=quantity,
                    is_prescription=med_data.get('is_prescription', False)
                )
                
                # Medication.save() automatically updates quantity
                # Now create inventory log
                InventoryLog.objects.create(
                    medication=medication,
                    log_type='CHECKOUT',
                    quantity_change=-quantity,
                    previous_quantity=previous_quantity,
                    new_quantity=medication.current_quantity,
                    performed_by=astronaut,
                    notes=f"Checkout by {astronaut.name}"
                )
                
                checkouts_created += 1
            
            # Unlock container
            unlock_success = send_esp32_unlock(astronaut)
            
            SystemLog.objects.create(
                event_type='CONTAINER_UNLOCK',
                astronaut=astronaut,
                description=f"Checkout completed: {checkouts_created} medications dispensed",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            response_data = {
                'success': True,
                'checkouts': checkouts_created,
                'unlock_status': unlock_success,
                # 'warnings': warnings_triggered
            }
            
            # if warnings_triggered:
            #     response_data['warning_message'] = f"Warning: Excessive medication withdrawal detected for {', '.join([w['medication'] for w in warnings_triggered])}"
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'POST required'}, status=400)


def check_medication_threshold(astronaut, medication, quantity):
    """Check if medication withdrawal exceeds thresholds"""
    try:
        threshold = medication.threshold
    except:
        return False, None
    
    # warning_created = False
    max_severity = None
    
    # Check single dose limit
    # if quantity > threshold.single_dose_limit:
    #     severity = 'CRITICAL' if quantity > threshold.single_dose_limit * 1.5 else 'HIGH'
    #     WarningLog.objects.create(
    #         astronaut=astronaut,
    #         medication=medication,
    #         quantity_taken=quantity,
    #         warning_message=f"Single dose limit exceeded: {quantity} units (limit: {threshold.single_dose_limit})",
    #         severity=severity
    #     )
    #     warning_created = True
    #     max_severity = severity
    
    # Check daily limit
    # today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # today_total = MedicationCheckout.objects.filter(
    #     astronaut=astronaut,
    #     medication=medication,
    #     checkout_time__gte=today_start
    # ).aggregate(total=Sum('quantity'))['total'] or 0
    # today_total += quantity
    
    # if today_total > threshold.daily_limit:
    #     WarningLog.objects.create(
    #         astronaut=astronaut,
    #         medication=medication,
    #         quantity_taken=quantity,
    #         warning_message=f"Daily limit exceeded: {today_total} units today (limit: {threshold.daily_limit})",
    #         severity='CRITICAL'
    #     )
    #     warning_created = True
    #     max_severity = 'CRITICAL'
    
    # return warning_created, max_severity


def send_esp32_unlock(astronaut):
    """Send unlock command to ESP32"""
    try:
        url = f"http://{ESP32_IP}/face-unlock"
        payload = {
            'username': astronaut.name,
            'user_id': str(astronaut.id)
        }
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error unlocking container: {e}")
        return False


# ============================================================================
# INVENTORY VIEWS
# ============================================================================

def inventory_dashboard(request):
    """Inventory dashboard"""
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
    """Medication detail view with transaction log"""
    medication = get_object_or_404(Medication, id=medication_id)
    
    thirty_days_ago = timezone.now() - timedelta(days=30)
    checkouts = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).order_by('-checkout_time')
    
    # Get transaction log (inventory logs)
    inventory_logs = InventoryLog.objects.filter(
        medication=medication
    ).order_by('-timestamp')[:50]  # Last 50 transactions
    
    daily_usage = MedicationCheckout.objects.filter(
        medication=medication,
        checkout_time__gte=thirty_days_ago
    ).annotate(
        date=TruncDate('checkout_time')
    ).values('date').annotate(
        total_quantity=Sum('quantity'),
        checkout_count=Count('id')
    ).order_by('date')
    
    total_dispensed_30d = sum(item['total_quantity'] for item in daily_usage)
    
    context = {
        'medication': medication,
        'checkouts': checkouts,
        'daily_usage': list(daily_usage),
        'inventory_logs': inventory_logs,
        'total_dispensed_30d': total_dispensed_30d
    }
    
    return render(request, 'medication_detail.html', context)


# ============================================================================
# ASTRONAUT MANAGEMENT (PROTECTED)
# ============================================================================

@login_required
def manage_astronauts(request):
    """Astronaut management page"""
    return render(request, 'manage_astronauts.html')


@csrf_exempt
@login_required
def add_astronaut(request):
    """Add new astronaut with face encoding"""
    if request.method == 'POST':
        try:
            astronaut_id = request.POST.get('astronaut_id')
            name = request.POST.get('name')
            photo = request.FILES.get('photo')
            password = request.POST.get('password', astronaut_id)
            
            if not all([astronaut_id, name, photo]):
                return JsonResponse({
                    'success': False,
                    'message': 'All fields are required'
                })
            
            # Create user account
            from django.contrib.auth.models import User
            user = User.objects.create_user(
                username=astronaut_id,
                password=password,
                first_name=name.split()[0] if name else '',
                last_name=' '.join(name.split()[1:]) if len(name.split()) > 1 else ''
            )
            
            # Create astronaut
            astronaut = Astronaut.objects.create(
                user=user,
                astronaut_id=astronaut_id,
                name=name,
                photo=photo
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
    """List all astronauts with photo URLs"""
    astronauts = Astronaut.objects.all()
    
    data = [{
        'id': a.id,
        'name': a.name,
        'astronaut_id': a.astronaut_id,
        'has_face_encoding': a.face_encoding is not None,
        'photo_url': a.photo.url if a.photo else None
    } for a in astronauts]
    
    return JsonResponse({'astronauts': data})


@csrf_exempt
@login_required
def update_astronaut_face(request):
    """Update astronaut face encoding with camera capture option"""
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
            
            # Update photo
            astronaut.photo = photo
            
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
@login_required
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
# MEDICATION MANAGEMENT (PROTECTED)
# ============================================================================

@login_required
def manage_medications(request):
    """Medication management page"""
    return render(request, 'manage_medications.html')


@csrf_exempt
@login_required
def add_medication(request):
    """Add new medication"""
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
@login_required
def update_medication_quantity(request):
    """Update medication quantity (for authorized users)"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            medication_id = data.get('medication_id')
            new_quantity = int(data.get('quantity'))
            reason = data.get('reason', 'Manual adjustment')
            
            medication = get_object_or_404(Medication, id=medication_id)
            previous_quantity = medication.current_quantity
            quantity_change = new_quantity - previous_quantity
            
            medication.current_quantity = new_quantity
            medication.save()
            
            # Log the change
            astronaut = request.user.astronaut if hasattr(request.user, 'astronaut') else None
            InventoryLog.objects.create(
                medication=medication,
                log_type='ADJUSTMENT',
                quantity_change=quantity_change,
                previous_quantity=previous_quantity,
                new_quantity=new_quantity,
                performed_by=astronaut,
                notes=reason
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Quantity updated successfully',
                'new_quantity': new_quantity
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
        'medication_type': m.medication_type,
        'medication_type_display': m.get_medication_type_display(),
        'dosage': m.dosage,
        'current_quantity': m.current_quantity,
        'minimum_quantity': m.minimum_quantity,
        'container_location': m.container_location,
        'status': m.status,
        'expiration_date': m.expiration_date.strftime('%Y-%m-%d') if m.expiration_date else None,
        'pill_image': m.pill_image.url if m.pill_image else None
    } for m in medications]
    
    return JsonResponse({'medications': data})


@csrf_exempt
@login_required
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
@login_required
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


# # ============================================================================
# # WARNING LOG
# # ============================================================================

# @login_required
# def warning_log_view(request):
#     """View medication warnings"""
#     warnings = WarningLog.objects.select_related('astronaut', 'medication', 'acknowledged_by').all()
    
#     # Filters
#     severity = request.GET.get('severity')
#     if severity:
#         warnings = warnings.filter(severity=severity)
    
#     acknowledged = request.GET.get('acknowledged')
#     if acknowledged == 'true':
#         warnings = warnings.filter(acknowledged=True)
#     elif acknowledged == 'false':
#         warnings = warnings.filter(acknowledged=False)
    
#     # Export to CSV if requested
#     if request.GET.get('export') == 'csv':
#         return export_warnings_csv(warnings)
    
#     stats = {
#         'total': warnings.count(),
#         'acknowledged': warnings.filter(acknowledged=True).count(),
#         'pending': warnings.filter(acknowledged=False).count(),
#         'critical': warnings.filter(severity='CRITICAL').count(),
#     }
    
#     return render(request, 'warning_log.html', {
#         'warnings': warnings[:100],
#         'stats': stats
#     })


# @login_required
# @require_POST
# def acknowledge_warning(request, warning_id):
#     """Acknowledge a warning"""
#     warning = get_object_or_404(WarningLog, id=warning_id)
    
#     astronaut, _ = Astronaut.objects.get_or_create(
#         user=request.user,
#         defaults={
#             'name': request.user.get_full_name() or request.user.username,
#             'astronaut_id': f'ADMIN_{request.user.id}'
#         }
#     )
    
#     warning.acknowledged = True
#     warning.acknowledged_by = astronaut
#     warning.acknowledged_at = timezone.now()
#     warning.save()
    
#     messages.success(request, "Warning acknowledged successfully.")
#     return redirect('medical_inventory:warning_log')


# def export_warnings_csv(warnings):
#     """Export warnings to CSV"""
#     response = HttpResponse(content_type='text/csv')
#     response['Content-Disposition'] = f'attachment; filename="warnings_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
#     writer = csv.writer(response)
#     writer.writerow([
#         'Date', 'Time', 'Astronaut', 'Medication', 'Quantity', 
#         'Severity', 'Message', 'Acknowledged', 'Acknowledged By', 'Acknowledged At'
#     ])
    
#     for warning in warnings:
#         writer.writerow([
#             warning.timestamp.strftime('%Y-%m-%d'),
#             warning.timestamp.strftime('%H:%M:%S'),
#             warning.astronaut.name,
#             warning.medication.name,
#             warning.quantity_taken,
#             warning.get_severity_display(),
#             warning.warning_message,
#             'Yes' if warning.acknowledged else 'No',
#             warning.acknowledged_by.name if warning.acknowledged_by else '',
#             warning.acknowledged_at.strftime('%Y-%m-%d %H:%M') if warning.acknowledged_at else ''
#         ])
    
#     return response


@login_required
def export_medications_csv(request):
    """Export medications to CSV"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="medications_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'ID', 'Name', 'Generic Name', 'Type', 'Dosage', 'Description',
        'Current Quantity', 'Minimum Quantity', 'Status', 'Location',
        'Expiration Date', 'Has Image'
    ])
    
    medications = Medication.objects.all()
    for med in medications:
        writer.writerow([
            med.id,
            med.name,
            med.generic_name,
            med.get_medication_type_display(),
            med.dosage,
            med.description,
            med.current_quantity,
            med.minimum_quantity,
            med.get_status_display(),
            med.container_location,
            med.expiration_date.strftime('%Y-%m-%d') if med.expiration_date else '',
            'Yes' if med.pill_image else 'No'
        ])
    
    return response


# ============================================================================
# PILL RECOGNITION
# ============================================================================

def pill_recognition(request):
    """Pill recognition page"""
    return render(request, 'pill_recognition.html')


@csrf_exempt
def recognize_pill(request):
    """Pill recognition endpoint"""
    if request.method == 'POST' and request.FILES.get('image'):
        return JsonResponse({
            'success': False,
            'message': 'Pill recognition model not yet implemented. Please upload training data and train the model.'
        })
    
    return JsonResponse({'error': 'No image provided'}, status=400)


# ===== ADMIN API ENDPOINTS =====

import base64
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
                photo.seek(0)  # Rewind file pointer after face_recognition consumed it
                astronaut.photo = base64.b64encode(photo.read()).decode('utf-8')  # Save the actual photo as base64
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
        # 'photo_url': a.photo.url if a.photo else None
        'photo_url': 'data:image/jpeg;base64,' + a.photo if a.photo else None
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

def inventory_dashboard(request):
    # Your existing view code
    medications = Medication.objects.all()
    total_medications = medications.count()
    low_stock_count = medications.filter(current_quantity__lte=10).count()
    # ... other context variables
    
    context = {
        'medications': medications,
        'total_medications': total_medications,
        'low_stock_count': low_stock_count,
        # ... other context
    }
    return render(request, 'inventory_dashboard.html', context)

def add_medication(request):
    if request.method == 'POST':
        form = MedicationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Medication added successfully!')
            return redirect('medical_inventory:inventory_dashboard')  # Adjust to your URL name
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = MedicationForm()
    
    return render(request, 'add_medication.html', {'form': form})

def medication_inventory_graph(request):
    return render(request, 'medication_line_graph.html')

@csrf_exempt
def medication_history_api(request):
    days = int(request.GET.get('days', 30))
    since = timezone.now() - timedelta(days=days) if days > 0 else None

    # Build list of every date in the range
    today = timezone.now().date()
    if since:
        start_date = since.date()
    else:
        first_log = InventoryLog.objects.order_by('timestamp').first()
        start_date = first_log.timestamp.date() if first_log else today

    all_dates = []
    current = start_date
    while current <= today:
        all_dates.append(current)
        current += timedelta(days=1)

    data = {}
    for med in Medication.objects.all():
        logs = InventoryLog.objects.filter(medication=med)
        if since:
            logs = logs.filter(timestamp__gte=since)
        logs = list(logs.order_by('timestamp'))

        if not logs:
            data[med.name] = {
                'type': med.get_medication_type_display(),
                'points': [
                    {'date': d.strftime('%Y-%m-%d'), 'quantity': med.current_quantity}
                    for d in all_dates
                ]
            }
            continue

        # Fill every date working backwards from current quantity
        log_map = {}
        for log in logs:
            log_map[log.timestamp.date()] = log.new_quantity

        points = []
        last_qty = med.current_quantity
        for d in reversed(all_dates):
            if d in log_map:
                last_qty = log_map[d]
            points.append({'date': d.strftime('%Y-%m-%d'), 'quantity': last_qty})

        points.reverse()

        data[med.name] = {
            'type': med.get_medication_type_display(),
            'points': points
        }

    summary = {
        'total':    Medication.objects.count(),
        'normal':   Medication.objects.filter(status='NORMAL').count(),
        'low':      Medication.objects.filter(status='LOW').count(),
        'critical': Medication.objects.filter(status__in=['CRITICAL', 'OUT']).count(),
    }
    return JsonResponse({'data': data, 'summary': summary})