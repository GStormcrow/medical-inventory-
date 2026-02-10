# Add these models to your medical_inventory/models.py file

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class WarningLog(models.Model):
    """Track warnings for excessive medication withdrawals"""
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='warning_logs')
    medication = models.ForeignKey('Medication', on_delete=models.CASCADE)
    quantity_taken = models.IntegerField()
    warning_message = models.TextField()
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default='MEDIUM')
    timestamp = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='acknowledged_warnings')
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['user', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.medication.name} - {self.timestamp}"
    
    def acknowledge(self, admin_user):
        """Mark warning as acknowledged by an admin"""
        self.acknowledged = True
        self.acknowledged_by = admin_user
        self.acknowledged_at = timezone.now()
        self.save()


class EmergencyAccess(models.Model):
    """Log emergency access to the medication system"""
    pin_used = models.CharField(max_length=255)  # Store hashed
    accessed_at = models.DateTimeField(auto_now_add=True)
    accessed_by = models.CharField(max_length=255, blank=True, null=True)  # If identifiable
    reason = models.TextField(blank=True, null=True)
    medications_accessed = models.ManyToManyField('Medication', blank=True)
    
    class Meta:
        ordering = ['-accessed_at']
    
    def __str__(self):
        return f"Emergency Access - {self.accessed_at}"


class AstronautProfile(models.Model):
    """Extended profile for astronauts with photo storage"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='astronaut_profile')
    photo = models.ImageField(upload_to='astronaut_photos/', blank=True, null=True)
    face_encoding = models.BinaryField(blank=True, null=True)  # Store facial recognition data
    medical_clearance = models.BooleanField(default=True)
    emergency_contact = models.CharField(max_length=255, blank=True)
    blood_type = models.CharField(max_length=5, blank=True)
    allergies = models.TextField(blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)
    registered_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='registered_astronauts')
    is_active_astronaut = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.user.username} - Astronaut Profile"


class MedicationThreshold(models.Model):
    """Define thresholds for medication warnings"""
    medication = models.OneToOneField('Medication', on_delete=models.CASCADE, related_name='threshold')
    daily_limit = models.IntegerField(help_text="Maximum units per day")
    single_dose_limit = models.IntegerField(help_text="Maximum units per single withdrawal")
    warning_percentage = models.IntegerField(default=80, help_text="Percentage of limit to trigger warning")
    
    def __str__(self):
        return f"Threshold for {self.medication.name}"
