# Add these URL patterns to your medical_inventory/urls.py file

from django.urls import path
from . import views

urlpatterns = [
    # ... your existing URLs ...
    
    # Search functionality
    path('medications/search/', views.search_medications, name='search_medications'),
    
    # Medication dispensing
    path('medications/<int:medication_id>/dispense/', views.dispense_medication, name='dispense_medication'),
    
    # Warning logs
    path('warnings/', views.warning_log_view, name='warning_log'),
    path('warnings/<int:warning_id>/acknowledge/', views.acknowledge_warning, name='acknowledge_warning'),
    
    # CSV Export
    path('medications/export/', views.export_medications_csv, name='export_medications_csv'),
    
    # Astronaut registration (admin only)
    path('astronauts/register/', views.register_astronaut, name='register_astronaut'),
    
    # Photo capture
    path('astronauts/capture-photo/', views.capture_astronaut_photo, name='capture_photo'),
    
    # Emergency access
    path('emergency-access/', views.emergency_access, name='emergency_access'),
]
