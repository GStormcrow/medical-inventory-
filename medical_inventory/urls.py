from django.urls import path

from nasa import settings
from . import views

from django.conf import settings
from django.conf.urls.static import static

app_name = 'medical_inventory'

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main pages
    path('', views.home, name='home'),
    path('lockscreen/', views.lockscreen, name='lockscreen'),
    path('medication-selection/<int:astronaut_id>/', views.medication_selection, name='medication_selection'),
    
    # Inventory pages
    path('inventory/', views.inventory_dashboard, name='inventory_dashboard'),
     path('<int:medication_id>/', views.medication_detail, name='medication_detail'),
    path('add/', views.add_medication, name='add_medication'),
    # path('inventory/<int:medication_id>/', views.medication_detail, name='medication_detail'),
    # path('<int:id>/', views.medication_detail, name='medication_detail'),
    # path('inventory/add/', views.add_medication, name='add_medication'),
    
    # Pill recognition
    path('pill-recognition/', views.pill_recognition, name='pill_recognition'),
    
    # Admin Management Pages (PROTECTED)
    path('manage/astronauts/', views.manage_astronauts, name='manage_astronauts'),
    path('manage/medications/', views.manage_medications, name='manage_medications'),
    
    # Warning System
    # path('warnings/', views.warning_log_view, name='warning_log'),
    # path('warnings/<int:warning_id>/acknowledge/', views.acknowledge_warning, name='acknowledge_warning'),
    
    # API endpoints - Face Authentication
    path('api/authenticate/', views.authenticate_face, name='authenticate_face'),
    path('api/checkout/', views.checkout_medication, name='checkout_medication'),
    path('api/recognize-pill/', views.recognize_pill, name='recognize_pill'),
    
    # API endpoints - Astronaut Management
    path('api/astronauts/add/', views.add_astronaut, name='add_astronaut'),
    path('api/astronauts/list/', views.list_astronauts, name='list_astronauts'),
    path('api/astronauts/update-face/', views.update_astronaut_face, name='update_astronaut_face'),
    path('api/astronauts/delete/<int:astronaut_id>/', views.delete_astronaut, name='delete_astronaut'),
    
    # API endpoints - Medication Management
    path('api/medications/add/', views.add_medication, name='add_medication'),
    path('api/medications/list/', views.list_medications, name='list_medications'),
    path('api/medications/update-image/', views.update_medication_image, name='update_medication_image'),
    path('api/medications/update-quantity/', views.update_medication_quantity, name='update_medication_quantity'),
    path('api/medications/delete/<int:medication_id>/', views.delete_medication, name='delete_medication'),
    
    path('inventory/graph/', views.medication_inventory_graph, name='inventory_graph'),
    path('api/medications/history/', views.medication_history_api, name='medication_history_api'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
