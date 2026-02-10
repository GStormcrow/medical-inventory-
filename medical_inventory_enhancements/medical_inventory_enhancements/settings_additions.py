# Add these settings to your nasa/settings.py

import hashlib
import os

# Emergency Access PIN Configuration
# For production, use a secure PIN and change this!
# To generate a hash: hashlib.sha256('YOUR_PIN'.encode()).hexdigest()
EMERGENCY_PIN_HASH = hashlib.sha256('EMERGENCY123'.encode()).hexdigest()

# ESP32 Configuration
ESP32_IP_ADDRESS = '192.168.1.100'  # UPDATE THIS to your ESP32's IP address

# For serial communication (alternative to WiFi)
ESP32_SERIAL_PORT = '/dev/ttyUSB0'  # Linux
# ESP32_SERIAL_PORT = 'COM3'  # Windows
ESP32_BAUD_RATE = 115200

# QR Code Configuration
QR_CODE_ERROR_CORRECTION = 'H'  # High error correction

# File Upload Settings
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Make sure these directories exist
os.makedirs(os.path.join(MEDIA_ROOT, 'astronaut_photos'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'pill_images'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'temp'), exist_ok=True)

# Installed Apps - Make sure these are included
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'medical_inventory',
]

# Middleware - Make sure CSRF is included
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# For development - serve media files
if DEBUG:
    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Database - Use PostgreSQL for production
# For development, SQLite is fine
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# For PostgreSQL (recommended for production):
"""
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'medical_inventory_db',
        'USER': 'your_username',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
"""

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'debug.log'),
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
        'medical_inventory': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
