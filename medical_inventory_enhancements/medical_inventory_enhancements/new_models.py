# Add these models to medical_inventory/models.py

from django.db import models
from django.utils import timezone

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


# Also register these in admin.py:
"""
from django.contrib import admin
from .models import WarningLog, MedicationThreshold, EmergencyAccess

@admin.register(WarningLog)
class WarningLogAdmin(admin.ModelAdmin):
    list_display = ['astronaut', 'medication', 'quantity_taken', 'severity', 'timestamp', 'acknowledged']
    list_filter = ['severity', 'acknowledged', 'timestamp']
    search_fields = ['astronaut__name', 'medication__name']
    date_hierarchy = 'timestamp'

@admin.register(MedicationThreshold)
class MedicationThresholdAdmin(admin.ModelAdmin):
    list_display = ['medication', 'daily_limit', 'single_dose_limit', 'warning_percentage']
    search_fields = ['medication__name']

@admin.register(EmergencyAccess)
class EmergencyAccessAdmin(admin.ModelAdmin):
    list_display = ['accessed_at', 'accessed_by_name', 'ip_address']
    list_filter = ['accessed_at']
    search_fields = ['accessed_by_name', 'reason']
    date_hierarchy = 'accessed_at'
    readonly_fields = ['accessed_at', 'pin_hash']
"""
