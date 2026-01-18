# urls.py (in your app directory)
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
    
    # API endpoints
    path('api/video-feed/', views.video_feed, name='video_feed'),
    path('api/authenticate/', views.authenticate_face, name='authenticate_face'),
    path('api/checkout/', views.checkout_medication, name='checkout_medication'),
    path('api/recognize-pill/', views.recognize_pill, name='recognize_pill'),
]