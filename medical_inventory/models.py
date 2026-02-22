# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Astronaut(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    astronaut_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    face_encoding = models.BinaryField(null=True, blank=True)  # Store face encoding
    # Add this to your Astronaut model:
    # photo = models.ImageField(upload_to='astronaut_photos/', null=True, blank=True)
    # photo = models.ImageField(upload_to='astronaut_photos/', null=True, blank=True)
    photo = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.astronaut_id})"


class Medication(models.Model):
    pill_shape = models.CharField(
        max_length=20,
        choices=[
            ('ROUND', 'Round'),
            ('OVAL', 'Oval'),
            ('CAPSULE', 'Capsule'),
            ('SQUARE', 'Square'),
            ('DIAMOND', 'Diamond'),
        ],
        blank=True,
        null=True
    )
    
    pill_color = models.CharField(
        max_length=20,
        choices=[
            ('WHITE', 'White'),
            ('RED', 'Red'),
            ('BLUE', 'Blue'),
            ('GREEN', 'Green'),
            ('YELLOW', 'Yellow'),
            ('ORANGE', 'Orange'),
            ('PINK', 'Pink'),
            ('BROWN', 'Brown'),
            ('BLACK', 'Black'),
            ('MULTI-COLOR', 'Multi-color'),
        ],
        blank=True,
        null=True
    )
    
    pill_imprint = models.CharField(
        max_length=50,
        blank=True,
        help_text="Text/numbers on pill"
    )
    
    pill_size = models.CharField(
        max_length=10,
        choices=[
            ('SMALL', 'Small (<10mm)'),
            ('MEDIUM', 'Medium (10-15mm)'),
            ('LARGE', 'Large (>15mm)'),
        ],
        blank=True,
        null=True
    )
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
        
class WarningLog(models.Model):
    """Track warnings for excessive medication withdrawals"""
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    astronaut = models.ForeignKey('Astronaut', on_delete=models.CASCADE, related_name='warning_logs')
    medication = models.ForeignKey('Medication', on_delete=models.CASCADE)
    quantity_taken = models.IntegerField()
    warning_message = models.TextField()
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='MEDIUM')
    timestamp = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey('Astronaut', on_delete=models.SET_NULL, null=True, blank=True, related_name='acknowledged_warnings')
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['astronaut', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.astronaut.name} - {self.medication.name} - {self.timestamp}"


class MedicationThreshold(models.Model):
    """Define thresholds for medication warnings"""
    medication = models.OneToOneField('Medication', on_delete=models.CASCADE, related_name='threshold')
    daily_limit = models.IntegerField(help_text="Maximum units per day")
    single_dose_limit = models.IntegerField(help_text="Maximum units per single withdrawal")
    warning_percentage = models.IntegerField(default=80, help_text="Percentage of limit to trigger warning")
    
    def __str__(self):
        return f"Threshold for {self.medication.name}"


class EmergencyAccess(models.Model):
    """Log emergency access to the medication system"""
    accessed_at = models.DateTimeField(auto_now_add=True)
    pin_hash = models.CharField(max_length=255)
    accessed_by_name = models.CharField(max_length=255, blank=True)
    reason = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    medications_accessed = models.TextField(blank=True)  # JSON string of medications
    
    class Meta:
        ordering = ['-accessed_at']
    
    def __str__(self):
        return f"Emergency Access - {self.accessed_at}"