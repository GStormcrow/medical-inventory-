# Add these views to your medical_inventory/views.py file

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Sum
from django.utils import timezone
from django.views.decorators.http import require_POST
from datetime import datetime, timedelta
import csv
import hashlib
import io
import base64
from PIL import Image

from .models import Medication, WarningLog, EmergencyAccess, AstronautProfile, MedicationThreshold


def is_admin(user):
    """Check if user is admin/superuser"""
    return user.is_superuser or user.is_staff


@login_required
def search_medications(request):
    """Search medications with AJAX support"""
    query = request.GET.get('q', '').strip()
    
    if not query:
        medications = Medication.objects.all()[:20]
    else:
        medications = Medication.objects.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(category__icontains=query)
        )[:20]
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Return JSON for AJAX requests
        data = [{
            'id': med.id,
            'name': med.name,
            'quantity': med.quantity,
            'description': med.description,
            'category': med.category if hasattr(med, 'category') else '',
        } for med in medications]
        return JsonResponse({'medications': data})
    
    return render(request, 'medical_inventory/medication_search.html', {
        'medications': medications,
        'query': query
    })


@login_required
def dispense_medication(request, medication_id):
    """Dispense medication with warning check"""
    medication = get_object_or_404(Medication, id=medication_id)
    
    if request.method == 'POST':
        quantity = int(request.POST.get('quantity', 1))
        
        # Check if quantity exceeds available stock
        if quantity > medication.quantity:
            messages.error(request, f"Insufficient stock. Only {medication.quantity} available.")
            return redirect('medication_detail', medication_id=medication_id)
        
        # Check thresholds and create warnings if needed
        warning_created = check_medication_threshold(request.user, medication, quantity)
        
        # Update inventory
        medication.quantity -= quantity
        medication.save()
        
        # Log the dispensing
        # Add your dispensing log here
        
        if warning_created:
            messages.warning(request, "Warning: Excessive medication withdrawal detected. Admin has been notified.")
        else:
            messages.success(request, f"Successfully dispensed {quantity} units of {medication.name}")
        
        return redirect('medication_list')
    
    return render(request, 'medical_inventory/dispense_medication.html', {
        'medication': medication
    })


def check_medication_threshold(user, medication, quantity):
    """Check if medication withdrawal exceeds thresholds and create warning if needed"""
    try:
        threshold = medication.threshold
    except MedicationThreshold.DoesNotExist:
        # No threshold set, no warning needed
        return False
    
    warning_created = False
    
    # Check single dose limit
    if quantity > threshold.single_dose_limit:
        severity = 'HIGH' if quantity > threshold.single_dose_limit * 1.5 else 'MEDIUM'
        WarningLog.objects.create(
            user=user,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Single dose limit exceeded: {quantity} units (limit: {threshold.single_dose_limit})",
            severity=severity
        )
        warning_created = True
    
    # Check daily limit
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_total = WarningLog.objects.filter(
        user=user,
        medication=medication,
        timestamp__gte=today_start
    ).aggregate(total=Sum('quantity_taken'))['total'] or 0
    
    today_total += quantity
    
    if today_total > threshold.daily_limit:
        WarningLog.objects.create(
            user=user,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Daily limit exceeded: {today_total} units today (limit: {threshold.daily_limit})",
            severity='CRITICAL'
        )
        warning_created = True
    elif today_total > threshold.daily_limit * (threshold.warning_percentage / 100):
        WarningLog.objects.create(
            user=user,
            medication=medication,
            quantity_taken=quantity,
            warning_message=f"Approaching daily limit: {today_total} units today (limit: {threshold.daily_limit})",
            severity='LOW'
        )
        warning_created = True
    
    return warning_created


@login_required
@user_passes_test(is_admin)
def warning_log_view(request):
    """View all medication warnings"""
    warnings = WarningLog.objects.select_related('user', 'medication', 'acknowledged_by').all()
    
    # Filter options
    severity = request.GET.get('severity')
    if severity:
        warnings = warnings.filter(severity=severity)
    
    acknowledged = request.GET.get('acknowledged')
    if acknowledged == 'true':
        warnings = warnings.filter(acknowledged=True)
    elif acknowledged == 'false':
        warnings = warnings.filter(acknowledged=False)
    
    return render(request, 'medical_inventory/warning_log.html', {
        'warnings': warnings
    })


@login_required
@user_passes_test(is_admin)
def acknowledge_warning(request, warning_id):
    """Acknowledge a warning"""
    warning = get_object_or_404(WarningLog, id=warning_id)
    warning.acknowledge(request.user)
    messages.success(request, "Warning acknowledged.")
    return redirect('warning_log')


@login_required
@user_passes_test(is_admin)
def export_medications_csv(request):
    """Export all medications to CSV"""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="medications_export_{}.csv"'.format(
        timezone.now().strftime('%Y%m%d_%H%M%S')
    )
    
    writer = csv.writer(response)
    writer.writerow(['ID', 'Name', 'Description', 'Quantity', 'Category', 'Expiration Date', 'Last Updated'])
    
    medications = Medication.objects.all()
    for med in medications:
        writer.writerow([
            med.id,
            med.name,
            getattr(med, 'description', ''),
            med.quantity,
            getattr(med, 'category', ''),
            getattr(med, 'expiration_date', ''),
            getattr(med, 'updated_at', '')
        ])
    
    return response


@login_required
@user_passes_test(is_admin)
def register_astronaut(request):
    """Register new astronaut (admin only)"""
    if request.method == 'POST':
        # Get user data
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Create user
        from django.contrib.auth.models import User
        user = User.objects.create_user(username=username, email=email, password=password)
        
        # Create astronaut profile
        profile = AstronautProfile.objects.create(
            user=user,
            registered_by=request.user,
            emergency_contact=request.POST.get('emergency_contact', ''),
            blood_type=request.POST.get('blood_type', ''),
            allergies=request.POST.get('allergies', '')
        )
        
        # Handle in-site photo capture
        photo_data = request.POST.get('photo_data')
        if photo_data:
            # Decode base64 image
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]
            
            from django.core.files.base import ContentFile
            profile.photo.save(
                f'{username}_photo.{ext}',
                ContentFile(base64.b64decode(imgstr)),
                save=True
            )
        
        messages.success(request, f"Astronaut {username} registered successfully!")
        return redirect('astronaut_list')
    
    return render(request, 'medical_inventory/register_astronaut.html')


@require_POST
def emergency_access(request):
    """Handle emergency access with PIN"""
    pin = request.POST.get('emergency_pin')
    
    # Hash the PIN for comparison
    hashed_pin = hashlib.sha256(pin.encode()).hexdigest()
    
    # Check against stored emergency PIN (you should store this in settings or database)
    from django.conf import settings
    correct_pin_hash = getattr(settings, 'EMERGENCY_PIN_HASH', None)
    
    if correct_pin_hash and hashed_pin == correct_pin_hash:
        # Log emergency access
        emergency_log = EmergencyAccess.objects.create(
            pin_used=hashed_pin,
            accessed_by=request.POST.get('name', 'Unknown'),
            reason=request.POST.get('reason', '')
        )
        
        # Grant temporary access - implement your lock control here
        # This could trigger ESP32 unlock
        
        messages.success(request, "Emergency access granted. Access logged.")
        return JsonResponse({'success': True, 'message': 'Emergency access granted'})
    else:
        messages.error(request, "Invalid emergency PIN.")
        return JsonResponse({'success': False, 'message': 'Invalid PIN'}, status=403)


@login_required
def capture_astronaut_photo(request):
    """Capture photo in-site for astronaut profile"""
    if request.method == 'POST':
        photo_data = request.POST.get('photo_data')
        profile_id = request.POST.get('profile_id')
        
        if photo_data and profile_id:
            profile = get_object_or_404(AstronautProfile, id=profile_id)
            
            # Decode base64 image
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]
            
            from django.core.files.base import ContentFile
            profile.photo.save(
                f'{profile.user.username}_photo.{ext}',
                ContentFile(base64.b64decode(imgstr)),
                save=True
            )
            
            return JsonResponse({'success': True})
        
        return JsonResponse({'success': False, 'error': 'Missing data'}, status=400)
    
    profile_id = request.GET.get('profile_id')
    return render(request, 'medical_inventory/capture_photo.html', {
        'profile_id': profile_id
    })
