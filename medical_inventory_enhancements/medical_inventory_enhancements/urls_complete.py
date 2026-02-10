# Enhanced urls.py - Add these routes to your medical_inventory/urls.py

from django.urls import path
from . import views

app_name = 'medical_inventory'

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('lockscreen/', views.lockscreen, name='lockscreen'),
    path('medication-selection/<int:astronaut_id>/', views.medication_selection, name='medication_selection'),
    
    # Inventory pages
    path('inventory/', views.inventory_dashboard, name='inventory_dashboard'),
    path('inventory/<int:medication_id>/', views.medication_detail, name='medication_detail'),
    
    # Pill recognition
    path('pill-recognition/', views.pill_recognition, name='pill_recognition'),
    
    # Admin Management Pages
    path('manage/astronauts/', views.manage_astronauts, name='manage_astronauts'),
    path('manage/medications/', views.manage_medications, name='manage_medications'),
    
    # NEW: Warning System
    path('warnings/', views.warning_log_view, name='warning_log'),
    path('warnings/<int:warning_id>/acknowledge/', views.acknowledge_warning, name='acknowledge_warning'),
    
    # NEW: Emergency Access
    path('emergency/', views.emergency_access, name='emergency_access'),
    
    # NEW: QR Code
    path('qr-code/', views.qr_code_page, name='qr_code_page'),
    path('qr-code/generate/', views.generate_qr_code, name='generate_qr_code'),
    
    # NEW: ESP32 Dashboard
    path('esp32/dashboard/', views.esp32_dashboard, name='esp32_dashboard'),
    
    # NEW: Photo Capture
    path('capture-photo/', views.capture_astronaut_photo, name='capture_photo'),
    
    # API endpoints
    path('api/authenticate/', views.authenticate_face, name='authenticate_face'),
    path('api/checkout/', views.checkout_medication, name='checkout_medication'),
    path('api/recognize-pill/', views.recognize_pill, name='recognize_pill'),
    
    # NEW: Search API
    path('api/medications/search/', views.search_medications_api, name='search_medications_api'),
    
    # NEW: CSV Export
    path('api/medications/export-csv/', views.export_medications_csv, name='export_medications_csv'),
    
    # Astronaut Admin APIs
    path('api/astronauts/add/', views.add_astronaut, name='add_astronaut'),
    path('api/astronauts/list/', views.list_astronauts, name='list_astronauts'),
    path('api/astronauts/update-face/', views.update_astronaut_face, name='update_astronaut_face'),
    path('api/astronauts/delete/<int:astronaut_id>/', views.delete_astronaut, name='delete_astronaut'),
    
    # Medication Admin APIs
    path('api/medications/add/', views.add_medication, name='add_medication'),
    path('api/medications/list/', views.list_medications, name='list_medications'),
    path('api/medications/update-image/', views.update_medication_image, name='update_medication_image'),
    path('api/medications/delete/<int:medication_id>/', views.delete_medication, name='delete_medication'),
]
