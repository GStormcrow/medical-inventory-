# NASA Medical Inventory - New Features Implementation Guide

## 📋 Overview
This guide covers the implementation of all requested features:
1. ✅ Search bar for medication selection and management
2. ✅ Warning system for excessive medication withdrawals
3. ✅ CSV export for medication database
4. ✅ In-site photo capture for astronauts
5. ✅ Admin-only astronaut registration
6. ✅ Emergency access with PIN
7. ✅ QR code generation for presentations
8. ✅ ESP32 WiFi connection
9. ✅ Database configuration

## 🚀 Step-by-Step Implementation

### Step 1: Update Database Models

1. Add the new models to `medical_inventory/models.py`:
   ```python
   # Copy contents from /home/claude/new_models.py
   # Add WarningLog, MedicationThreshold, and EmergencyAccess models
   ```

2. Create and run migrations:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

3. Register new models in admin.py:
   ```python
   # Copy the admin registrations from new_models.py
   ```

### Step 2: Update Views

1. Replace or merge your `medical_inventory/views.py` with:
   ```python
   # Copy contents from /home/claude/enhanced_views.py
   # This includes all new functionality
   ```

### Step 3: Update URLs

1. Update `medical_inventory/urls.py`:
   ```python
   # Copy contents from /home/claude/urls_complete.py
   ```

### Step 4: Update Settings

1. Add to `nasa/settings.py`:
   ```python
   # Copy contents from /home/claude/settings_additions.py
   ```

2. Configure ESP32 IP address:
   ```python
   ESP32_IP_ADDRESS = '192.168.1.XXX'  # Your ESP32's IP
   ```

3. Set emergency PIN (change from default!):
   ```python
   import hashlib
   EMERGENCY_PIN_HASH = hashlib.sha256('YOUR_SECRET_PIN'.encode()).hexdigest()
   ```

### Step 5: Update Templates

Replace existing templates with enhanced versions:

1. **Medication Selection (with search)**:
   - Copy `/home/claude/templates/medication_selection_enhanced.html`
   - Rename to `medical_inventory/templates/medication_selection.html`

2. **Manage Medications (with search & CSV export)**:
   - Copy `/home/claude/templates/manage_medications_enhanced.html`
   - Rename to `medical_inventory/templates/manage_medications.html`

3. **New Templates** (add these):
   - `/home/claude/templates/warning_log_new.html` → `warning_log.html`
   - `/home/claude/templates/emergency_access.html`
   - `/home/claude/templates/qr_code.html`
   - `/home/claude/templates/capture_photo.html` (from earlier output)

### Step 6: Update Navigation

Update `base.html` navigation to include new pages:

```html
<ul class="nav-links" id="navLinks">
    <li><a href="{% url 'medical_inventory:home' %}">Home</a></li>
    <li><a href="{% url 'medical_inventory:lockscreen' %}">Access System</a></li>
    <li><a href="{% url 'medical_inventory:inventory_dashboard' %}">Inventory</a></li>
    <li><a href="{% url 'medical_inventory:warning_log' %}">Warnings</a></li>
    <li><a href="{% url 'medical_inventory:emergency_access' %}">Emergency</a></li>
    <li><a href="{% url 'medical_inventory:qr_code_page' %}">QR Code</a></li>
    <li><a href="{% url 'medical_inventory:manage_astronauts' %}">Astronauts</a></li>
    <li><a href="{% url 'medical_inventory:manage_medications' %}">Medications</a></li>
    <li><a href="/admin/" target="_blank">Admin</a></li>
</ul>
```

### Step 7: Install Dependencies

Update `requirements.txt`:

```txt
# Core
Django>=4.2.0
Pillow>=10.0.0

# Face Recognition
face-recognition>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0

# QR Code Generation
qrcode[pil]>=7.4.2

# Other
requests>=2.31.0
psycopg2-binary>=2.9.6  # For PostgreSQL
python-dotenv>=1.0.0
```

Install:
```bash
pip install -r requirements.txt
```

### Step 8: ESP32 Configuration

1. **Upload Arduino code** to ESP32:
   - Open `hardware/esp32_lock_controller.ino` in Arduino IDE
   - Update WiFi credentials:
     ```cpp
     const char* ssid = "YOUR_WIFI_SSID";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```
   - Upload to ESP32

2. **Find ESP32 IP address**:
   - Check Serial Monitor after upload
   - Update in Django settings.py

3. **Test connection**:
   ```python
   # In Django shell
   from medical_inventory.views import check_esp32_status
   status = check_esp32_status()
   print(status)
   ```

### Step 9: Create Sample Data

1. **Create medication thresholds** (in Django admin or shell):
   ```python
   from medical_inventory.models import Medication, MedicationThreshold
   
   # For each medication
   med = Medication.objects.get(name='Ibuprofen')
   MedicationThreshold.objects.create(
       medication=med,
       daily_limit=6,  # Max 6 pills per day
       single_dose_limit=2,  # Max 2 pills at once
       warning_percentage=80  # Warn at 80% of limit
   )
   ```

2. **Create admin user** (if not exists):
   ```bash
   python manage.py createsuperuser
   ```

### Step 10: Testing Each Feature

#### Test 1: Medication Search
- Go to medication selection page
- Type in search box
- Verify real-time filtering works

#### Test 2: Warning System
- Set up medication threshold
- Attempt to checkout excessive quantity
- Check warning log at `/warnings/`
- Verify warning appears and can be acknowledged

#### Test 3: CSV Export
- Go to manage medications
- Click "Export to CSV" button
- Verify file downloads with all medication data

#### Test 4: In-Site Photo Capture
- Go to manage astronauts
- Click "Add Astronaut"
- Use webcam to capture photo
- Verify face encoding is created

#### Test 5: Emergency Access
- Go to `/emergency/`
- Enter default PIN: EMERGENCY123
- Verify container unlocks (if ESP32 connected)

#### Test 6: QR Code
- Go to `/qr-code/`
- Verify QR code displays
- Test download and print functions
- Scan with phone to test

#### Test 7: ESP32 Connection
- Verify ESP32 is powered and connected to WiFi
- Check Django logs for connection status
- Test unlock from medication checkout

## 🔧 Troubleshooting

### Issue: Migrations fail
```bash
# Delete db.sqlite3 and start fresh
rm db.sqlite3
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### Issue: ESP32 not connecting
1. Check ESP32 Serial Monitor for IP address
2. Verify ESP32 and Django server are on same network
3. Update ESP32_IP_ADDRESS in settings.py
4. Test with: `curl http://ESP32_IP/status`

### Issue: Face recognition not working
```bash
# Install dlib dependencies (Linux)
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib face-recognition
```

### Issue: QR code not generating
```bash
pip install qrcode[pil] Pillow
```

## 📊 Database Schema

New tables added:
- `medical_inventory_warninglog` - Stores medication warnings
- `medical_inventory_medicationthreshold` - Defines limits for medications
- `medical_inventory_emergencyaccess` - Logs emergency access events

## 🔐 Security Notes

1. **Change default emergency PIN!**
   ```python
   # In settings.py
   EMERGENCY_PIN_HASH = hashlib.sha256('YOUR_SECURE_PIN'.encode()).hexdigest()
   ```

2. **For production deployment**:
   - Use PostgreSQL instead of SQLite
   - Set DEBUG = False
   - Use environment variables for secrets
   - Enable HTTPS
   - Configure CORS properly

3. **Admin access restriction**:
   - Only superusers can register astronauts
   - Only staff/superusers can view warning log
   - Emergency access is logged

## 🌐 Free Hosting Options

1. **PythonAnywhere** (Recommended for beginners)
   - Free tier available
   - Easy Django deployment
   - https://www.pythonanywhere.com

2. **Heroku**
   - Free tier with some limitations
   - Good for demos
   - https://www.heroku.com

3. **Railway**
   - Modern platform
   - Free starter plan
   - https://railway.app

4. **Render**
   - Free for web services
   - Auto-deploy from GitHub
   - https://render.com

### Deployment Steps (PythonAnywhere):

1. Create account at pythonanywhere.com
2. Upload your code via GitHub or zip file
3. Set up virtual environment
4. Configure WSGI file
5. Set up static files
6. Run migrations
7. Create superuser

## ✅ Feature Checklist

- [x] Search functionality in medication selection
- [x] Search functionality in medication management
- [x] Warning system for excessive withdrawals
- [x] Warning log with filtering and export
- [x] CSV export for medications
- [x] In-site photo capture for astronauts
- [x] Admin-only astronaut registration
- [x] Emergency PIN access
- [x] QR code generation and download
- [x] ESP32 WiFi connection
- [x] Database models and migrations

## 📞 Support

If you encounter issues:
1. Check the Django debug log: `debug.log`
2. Check ESP32 Serial Monitor
3. Verify all migrations are applied
4. Test individual components separately

## 🎉 You're Done!

Your NASA Medical Inventory system now has all requested features implemented!
