# Enhanced views.py - Add these functions to your existing medical_inventory/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Sum
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from datetime import datetime, timedelta
import csv
import hashlib
import io
import base64
import qrcode
import requests
import json
from PIL import Image

from .models import (
    Medication, Astronaut, MedicationCheckout, 
    SystemLog, InventoryLog, Prescription
)

# ============================================================================
# NEW MODELS - Add these to your models.py
# ============================================================================
"""
Add these new models to medical_inventory/models.py:

class WarningLog(models.Model):
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    astronaut = models.ForeignKey(Astronaut, on_delete=models.CASCADE, related_name='warning_logs')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    quantity_taken = models.IntegerField()
    warning_message = models.TextField()
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='MEDIUM')
    timestamp = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(Astronaut, on_delete=models.SET_NULL, null=True, blank=True, related_name='acknowledged_warnings')
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.astronaut.name} - {self.medication.name} - {self.timestamp}"

class MedicationThreshold(models.Model):
    medication = models.OneToOneField(Medication, on_delete=models.CASCADE, related_name='threshold')
    daily_limit = models.IntegerField(help_text="Maximum units per day")
    single_dose_limit = models.IntegerField(help_text="Maximum units per single withdrawal")
    warning_percentage = models.IntegerField(default=80, help_text="Percentage of limit to trigger warning")
    
    def __str__(self):
        return f"Threshold for {self.medication.name}"

class EmergencyAccess(models.Model):
    accessed_at = models.DateTimeField(auto_now_add=True)
    pin_hash = models.CharField(max_length=255)
    accessed_by_name = models.CharField(max_length=255, blank=True)
    reason = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    medications_accessed = models.TextField(blank=True)  # JSON string
    
    class Meta:
        ordering = ['-accessed_at']
    
    def __str__(self):
        return f"Emergency Access - {self.accessed_at}"
"""

# Import the new models (add after creating them)
# from .models import WarningLog, MedicationThreshold, EmergencyAccess


# ============================================================================
# SEARCH FUNCTIONALITY
# ============================================================================

@login_required
def search_medications_api(request):
    """API endpoint for medication search with AJAX support"""
    query = request.GET.get('q', '').strip()
    context = request.GET.get('context', 'selection')  # 'selection' or 'management'
    
    if not query:
        medications = Medication.objects.all()[:20]
    else:
        medications = Medication.objects.filter(
            Q(name__icontains=query) |
            Q(generic_name__icontains=query) |
            Q(medication_type__icontains=query) |
            Q(description__icontains=query)
        ).order_by('name')
    
    # Return JSON for AJAX
    data = [{
        'id': med.id,
        'name': med.name,
        'generic_name': med.generic_name,
        'quantity': med.current_quantity,
        'minimum_quantity': med.minimum_quantity,
        'medication_type': med.get_medication_type_display(),
        'dosage': med.dosage,
        'status': med.status,
        'location': med.container_location,
        'expiration_date': med.expiration_date.strftime('%Y-%m-%d') if med.expiration_date else None,
    } for med in medications]
    
    return JsonResponse({'medications': data})


# ============================================================================
# WARNING SYSTEM
# ============================================================================

def check_medication_threshold(astronaut, medication, quantity):
    """
    Check if medication withdrawal exceeds thresholds and create warning if needed
    Returns: (warning_created, warning_severity)
    """
    try:
        from .models import MedicationThreshold, WarningLog
    except ImportError:
        return False, None
    
    try:
        threshold = medication.threshold
    except:
        # No threshold set for this medication
        return False, None
    
    warning_created = False
    max_severity = None
    
    # Check 1: Single dose limit
    if quantity > threshold.single_dose_limit:
        severity = 'CRITICAL' if quantity > threshold.single_dose_limit * 1.5 else 'HIGH'
        WarningLog.objects.create(
            astronaut=astronaut,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Single dose limit exceeded: {quantity} units (limit: {threshold.single_dose_limit})",
            severity=severity
        )
        warning_created = True
        max_severity = severity
    
    # Check 2: Daily limit
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_checkouts = MedicationCheckout.objects.filter(
        astronaut=astronaut,
        medication=medication,
        checkout_time__gte=today_start
    )
    today_total = today_checkouts.aggregate(total=Sum('quantity'))['total'] or 0
    today_total += quantity
    
    if today_total > threshold.daily_limit:
        WarningLog.objects.create(
            astronaut=astronaut,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Daily limit exceeded: {today_total} units today (limit: {threshold.daily_limit})",
            severity='CRITICAL'
        )
        warning_created = True
        if not max_severity or max_severity != 'CRITICAL':
            max_severity = 'CRITICAL'
    
    elif today_total >= threshold.daily_limit * (threshold.warning_percentage / 100):
        WarningLog.objects.create(
            astronaut=astronaut,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Approaching daily limit: {today_total} units today (limit: {threshold.daily_limit})",
            severity='MEDIUM'
        )
        warning_created = True
        if not max_severity:
            max_severity = 'MEDIUM'
    
    return warning_created, max_severity


@login_required
@user_passes_test(lambda u: u.is_staff or u.is_superuser)
def warning_log_view(request):
    """View all medication warnings"""
    try:
        from .models import WarningLog
    except ImportError:
        messages.error(request, "Warning system not configured. Please add WarningLog model.")
        return redirect('medical_inventory:home')
    
    warnings = WarningLog.objects.select_related('astronaut', 'medication', 'acknowledged_by').all()
    
    # Filters
    severity = request.GET.get('severity')
    if severity:
        warnings = warnings.filter(severity=severity)
    
    acknowledged = request.GET.get('acknowledged')
    if acknowledged == 'true':
        warnings = warnings.filter(acknowledged=True)
    elif acknowledged == 'false':
        warnings = warnings.filter(acknowledged=False)
    
    # Export to CSV if requested
    if request.GET.get('export') == 'csv':
        return export_warnings_csv(warnings)
    
    # Statistics
    stats = {
        'total': warnings.count(),
        'acknowledged': warnings.filter(acknowledged=True).count(),
        'pending': warnings.filter(acknowledged=False).count(),
        'critical': warnings.filter(severity='CRITICAL').count(),
    }
    
    return render(request, 'medical_inventory/warning_log.html', {
        'warnings': warnings[:100],  # Limit to recent 100
        'stats': stats
    })


@login_required
@user_passes_test(lambda u: u.is_staff or u.is_superuser)
@require_POST
def acknowledge_warning(request, warning_id):
    """Acknowledge a warning"""
    try:
        from .models import WarningLog
        warning = get_object_or_404(WarningLog, id=warning_id)
        
        # Get or create an Astronaut object for the current user
        astronaut, _ = Astronaut.objects.get_or_create(
            user=request.user,
            defaults={
                'name': request.user.get_full_name() or request.user.username,
                'astronaut_id': f'ADMIN_{request.user.id}'
            }
        )
        
        warning.acknowledged = True
        warning.acknowledged_by = astronaut
        warning.acknowledged_at = timezone.now()
        warning.save()
        
        messages.success(request, "Warning acknowledged successfully.")
    except ImportError:
        messages.error(request, "Warning system not configured.")
    
    return redirect('medical_inventory:warning_log')


def export_warnings_csv(warnings):
    """Export warnings to CSV"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="warnings_export_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Date', 'Time', 'Astronaut', 'Medication', 'Quantity', 
        'Severity', 'Message', 'Acknowledged', 'Acknowledged By', 'Acknowledged At'
    ])
    
    for warning in warnings:
        writer.writerow([
            warning.timestamp.strftime('%Y-%m-%d'),
            warning.timestamp.strftime('%H:%M:%S'),
            warning.astronaut.name,
            warning.medication.name,
            warning.quantity_taken,
            warning.get_severity_display(),
            warning.warning_message,
            'Yes' if warning.acknowledged else 'No',
            warning.acknowledged_by.name if warning.acknowledged_by else '',
            warning.acknowledged_at.strftime('%Y-%m-%d %H:%M') if warning.acknowledged_at else ''
        ])
    
    return response


# ============================================================================
# CSV EXPORT FOR MEDICATIONS
# ============================================================================

@login_required
@user_passes_test(lambda u: u.is_staff or u.is_superuser)
def export_medications_csv(request):
    """Export all medications to CSV"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="medications_export_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
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
# IN-SITE PHOTO CAPTURE FOR ASTRONAUTS
# ============================================================================

@login_required
@user_passes_test(lambda u: u.is_staff or u.is_superuser)
def capture_astronaut_photo(request):
    """Capture photo in-site for astronaut registration"""
    if request.method == 'POST':
        photo_data = request.POST.get('photo_data')
        astronaut_id = request.POST.get('astronaut_id')
        
        if not photo_data:
            return JsonResponse({'success': False, 'error': 'No photo data'}, status=400)
        
        try:
            # Decode base64 image
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]
            
            from django.core.files.base import ContentFile
            import face_recognition
            import pickle
            
            # Decode image
            image_data = base64.b64decode(imgstr)
            
            if astronaut_id:
                # Update existing astronaut
                astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            else:
                # Create new astronaut (should have other data from form)
                name = request.POST.get('name')
                astro_id = request.POST.get('astronaut_id_code')
                
                if not name or not astro_id:
                    return JsonResponse({'success': False, 'error': 'Missing astronaut data'}, status=400)
                
                # Create user
                from django.contrib.auth.models import User
                username = astro_id.lower().replace(' ', '_')
                user = User.objects.create_user(
                    username=username,
                    password=request.POST.get('password', 'defaultpassword123')
                )
                
                astronaut = Astronaut.objects.create(
                    user=user,
                    name=name,
                    astronaut_id=astro_id
                )
            
            # Save photo
            astronaut.user.username  # Access to get the username
            photo_filename = f'{astronaut.astronaut_id}_photo.{ext}'
            astronaut.face_encoding = None  # Will be set below
            
            # Create face encoding
            # First save the image temporarily to process it
            image = Image.open(io.BytesIO(image_data))
            image_np = face_recognition.load_image_file(io.BytesIO(image_data))
            
            face_encodings = face_recognition.face_encodings(image_np)
            
            if not face_encodings:
                return JsonResponse({'success': False, 'error': 'No face detected in photo'}, status=400)
            
            # Save encoding
            astronaut.face_encoding = pickle.dumps(face_encodings[0])
            astronaut.save()
            
            # Now save the actual photo file
            # Note: You'll need to create a proper file path
            # This is a simplified version
            from django.core.files.uploadedfile import InMemoryUploadedFile
            image_io = io.BytesIO(image_data)
            uploaded_file = InMemoryUploadedFile(
                image_io, None, photo_filename, f'image/{ext}',
                len(image_data), None
            )
            
            # If your Astronaut model has a photo field, save it
            # astronaut.photo.save(photo_filename, uploaded_file, save=True)
            
            SystemLog.objects.create(
                event_type='AUTH_SUCCESS',
                astronaut=astronaut,
                description=f"Face encoding registered for {astronaut.name}",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            return JsonResponse({
                'success': True,
                'astronaut_id': astronaut.id,
                'message': 'Photo captured and face encoding created successfully'
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    # GET request - show capture interface
    astronaut_id = request.GET.get('astronaut_id')
    return render(request, 'medical_inventory/capture_photo.html', {
        'astronaut_id': astronaut_id
    })


# ============================================================================
# EMERGENCY ACCESS
# ============================================================================

# Store emergency PIN hash in settings.py:
# EMERGENCY_PIN_HASH = hashlib.sha256('your_emergency_pin'.encode()).hexdigest()

@csrf_exempt
def emergency_access(request):
    """Handle emergency access with PIN"""
    if request.method == 'POST':
        pin = request.POST.get('emergency_pin')
        name = request.POST.get('name', 'Unknown')
        reason = request.POST.get('reason', '')
        
        if not pin:
            return JsonResponse({'success': False, 'message': 'PIN required'}, status=400)
        
        # Hash the entered PIN
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()
        
        # Check against stored PIN (from settings)
        correct_pin_hash = getattr(settings, 'EMERGENCY_PIN_HASH', None)
        
        if not correct_pin_hash:
            # Default emergency PIN for development: "EMERGENCY123"
            correct_pin_hash = hashlib.sha256('EMERGENCY123'.encode()).hexdigest()
        
        if pin_hash == correct_pin_hash:
            try:
                # Log emergency access
                from .models import EmergencyAccess
                EmergencyAccess.objects.create(
                    pin_hash=pin_hash,
                    accessed_by_name=name,
                    reason=reason,
                    ip_address=request.META.get('REMOTE_ADDR')
                )
            except:
                pass
            
            # Send unlock signal to ESP32
            unlock_success = send_esp32_unlock()
            
            SystemLog.objects.create(
                event_type='CONTAINER_UNLOCK',
                description=f"Emergency access granted to: {name}. Reason: {reason}",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Emergency access granted',
                'unlock_status': unlock_success
            })
        else:
            SystemLog.objects.create(
                event_type='AUTH_FAILURE',
                description=f"Failed emergency access attempt by: {name}",
                ip_address=request.META.get('REMOTE_ADDR')
            )
            
            return JsonResponse({
                'success': False,
                'message': 'Invalid PIN'
            }, status=403)
    
    # GET request - show emergency access form
    return render(request, 'medical_inventory/emergency_access.html')


# ============================================================================
# QR CODE GENERATION
# ============================================================================

def generate_qr_code(request):
    """Generate QR code for the website"""
    # Get the full URL of your site
    site_url = request.build_absolute_uri('/')
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(site_url)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Return as downloadable image
    response = HttpResponse(buffer.getvalue(), content_type='image/png')
    response['Content-Disposition'] = f'attachment; filename="nasa_medical_inventory_qr.png"'
    
    return response


def qr_code_page(request):
    """Display QR code page for presentations"""
    site_url = request.build_absolute_uri('/')
    
    return render(request, 'medical_inventory/qr_code.html', {
        'site_url': site_url
    })


# ============================================================================
# ESP32 COMMUNICATION (WiFi Connection)
# ============================================================================

def send_esp32_unlock():
    """Send unlock command to ESP32 via HTTP"""
    esp32_ip = getattr(settings, 'ESP32_IP_ADDRESS', '192.168.1.100')
    
    try:
        # Send unlock request to ESP32
        response = requests.post(
            f'http://{esp32_ip}/unlock',
            json={
                'timestamp': timezone.now().isoformat(),
                'source': 'django_server'
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('success', False)
        else:
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")
        return False


def check_esp32_status():
    """Check if ESP32 is online and get status"""
    esp32_ip = getattr(settings, 'ESP32_IP_ADDRESS', '192.168.1.100')
    
    try:
        response = requests.get(f'http://{esp32_ip}/status', timeout=3)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'online': False, 'error': 'Bad response'}
            
    except requests.exceptions.RequestException as e:
        return {'online': False, 'error': str(e)}


@login_required
@user_passes_test(lambda u: u.is_staff or u.is_superuser)
def esp32_dashboard(request):
    """Admin dashboard to monitor ESP32 connection"""
    status = check_esp32_status()
    
    # Test unlock button
    if request.method == 'POST' and request.POST.get('action') == 'test_unlock':
        unlock_result = send_esp32_unlock()
        messages.success(request, f"Unlock test {'successful' if unlock_result else 'failed'}")
        return redirect('medical_inventory:esp32_dashboard')
    
    return render(request, 'medical_inventory/esp32_dashboard.html', {
        'esp32_status': status
    })


# ============================================================================
# ENHANCED CHECKOUT WITH WARNING CHECKS
# ============================================================================

@csrf_exempt
def checkout_medication(request):
    """Enhanced checkout with threshold checking"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            astronaut_id = data.get('astronaut_id')
            medications = data.get('medications', [])
            
            astronaut = get_object_or_404(Astronaut, id=astronaut_id)
            
            checkouts_created = 0
            warnings_triggered = []
            
            for med_data in medications:
                medication = get_object_or_404(Medication, id=med_data['medication_id'])
                quantity = med_data['quantity']
                
                # Check quantity available
                if quantity > medication.current_quantity:
                    return JsonResponse({
                        'success': False,
                        'message': f'Insufficient stock for {medication.name}'
                    }, status=400)
                
                # Check thresholds and create warnings
                warning_created, severity = check_medication_threshold(astronaut, medication, quantity)
                
                if warning_created:
                    warnings_triggered.append({
                        'medication': medication.name,
                        'severity': severity
                    })
                
                # Create checkout
                MedicationCheckout.objects.create(
                    astronaut=astronaut,
                    medication=medication,
                    quantity=quantity,
                    is_prescription=med_data.get('is_prescription', False)
                )
                
                checkouts_created += 1
            
            # Send unlock signal to ESP32
            unlock_success = send_esp32_unlock()
            
            # Log the event
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
                'warnings': warnings_triggered
            }
            
            if warnings_triggered:
                response_data['warning_message'] = f"Warning: Excessive medication withdrawal detected for {', '.join([w['medication'] for w in warnings_triggered])}"
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'POST required'}, status=400)
