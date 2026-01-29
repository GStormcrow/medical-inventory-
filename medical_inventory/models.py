# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Astronaut(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    astronaut_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    face_encoding = models.BinaryField(null=True, blank=True)  # Store face encoding
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.astronaut_id})"


class Medication(models.Model):
    MEDICATION_TYPES = [
        ('ANALGESIC', 'Pain Relief'),
        ('ANTIBIOTIC', 'Antibiotic'),
        ('ANTINAUSEA', 'Anti-Nausea'),
        ('SLEEP_AID', 'Sleep Aid'),
        ('ALLERGY', 'Allergy'),
        ('STIMULANT', 'Stimulant'),
        ('OTHER', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('NORMAL', 'Normal'),
        ('LOW', 'Low Stock'),
        ('CRITICAL', 'Critical'),
        ('OUT', 'Out of Stock'),
    ]
    
    name = models.CharField(max_length=200)
    current_quantity = models.IntegerField(default=0)
    minimum_quantity = models.IntegerField(default=10)
    expiration_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='NORMAL')
    
    # Keep these for backward compatibility but make optional
    generic_name = models.CharField(max_length=200, blank=True)
    medication_type = models.CharField(max_length=20, choices=MEDICATION_TYPES, default='OTHER')
    dosage = models.CharField(max_length=100, default='Standard')
    description = models.TextField(blank=True)
    container_location = models.CharField(max_length=50, default='A1')
    pill_image = models.ImageField(upload_to='pill_images/', null=True, blank=True)
    
    def __str__(self):
        return f"{self.name}"
    
    @property
    def is_low_stock(self):
        return self.current_quantity <= self.minimum_quantity
    
    def update_status(self):
        """Automatically update status based on quantity"""
        if self.current_quantity == 0:
            self.status = 'OUT'
        elif self.current_quantity <= (self.minimum_quantity * 0.5):  # 50% of minimum
            self.status = 'CRITICAL'
        elif self.current_quantity <= self.minimum_quantity:
            self.status = 'LOW'
        else:
            self.status = 'NORMAL'
    
    def save(self, *args, **kwargs):
        self.update_status()
        super().save(*args, **kwargs)


class Prescription(models.Model):
    astronaut = models.ForeignKey(Astronaut, on_delete=models.CASCADE, related_name='prescriptions')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    prescribed_dosage = models.CharField(max_length=100)
    frequency = models.CharField(max_length=100)  # e.g., "2x daily", "as needed"
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.astronaut.name} - {self.medication.name}"


class MedicationCheckout(models.Model):
    astronaut = models.ForeignKey(Astronaut, on_delete=models.CASCADE, related_name='checkouts')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=1)
    checkout_time = models.DateTimeField(default=timezone.now)
    is_prescription = models.BooleanField(default=False)  # Was this a prescribed med or additional?
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.astronaut.name} - {self.medication.name} - {self.checkout_time}"
    
    def save(self, *args, **kwargs):
        # Update medication inventory when checkout is saved
        if self.pk is None:  # Only on creation
            self.medication.current_quantity -= self.quantity
            self.medication.save()
        super().save(*args, **kwargs)


class InventoryLog(models.Model):
    LOG_TYPES = [
        ('CHECKOUT', 'Medication Checkout'),
        ('RESTOCK', 'Inventory Restock'),
        ('EXPIRED', 'Expired Medication Removed'),
        ('ADJUSTMENT', 'Manual Adjustment'),
    ]
    
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name='logs')
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    quantity_change = models.IntegerField()  # Positive for additions, negative for removals
    previous_quantity = models.IntegerField()
    new_quantity = models.IntegerField()
    timestamp = models.DateTimeField(default=timezone.now)
    performed_by = models.ForeignKey(Astronaut, on_delete=models.SET_NULL, null=True, blank=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.medication.name} - {self.log_type} - {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']


class SystemLog(models.Model):
    EVENT_TYPES = [
        ('AUTH_SUCCESS', 'Authentication Success'),
        ('AUTH_FAILURE', 'Authentication Failure'),
        ('CONTAINER_UNLOCK', 'Container Unlocked'),
        ('PILL_RECOGNITION', 'Pill Recognition Attempt'),
        ('SYSTEM_ERROR', 'System Error'),
    ]
    
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    astronaut = models.ForeignKey(Astronaut, on_delete=models.SET_NULL, null=True, blank=True)
    description = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.event_type} - {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']