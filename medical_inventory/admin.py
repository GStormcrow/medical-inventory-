from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Astronaut, Medication, Prescription, MedicationCheckout, InventoryLog, SystemLog, WarningLog, MedicationThreshold, EmergencyAccess

@admin.register(Astronaut)
class AstronautAdmin(admin.ModelAdmin):
    list_display = ['name', 'astronaut_id', 'created_at']
    search_fields = ['name', 'astronaut_id']

@admin.register(Medication)
class MedicationAdmin(admin.ModelAdmin):
    list_display = ['name', 'medication_type', 'dosage', 'current_quantity', 'minimum_quantity', 'is_low_stock']
    list_filter = ['medication_type']
    search_fields = ['name', 'generic_name']

@admin.register(Prescription)
class PrescriptionAdmin(admin.ModelAdmin):
    list_display = ['astronaut', 'medication', 'prescribed_dosage', 'frequency', 'is_active']
    list_filter = ['is_active']
    search_fields = ['astronaut__name', 'medication__name']

@admin.register(MedicationCheckout)
class MedicationCheckoutAdmin(admin.ModelAdmin):
    list_display = ['astronaut', 'medication', 'quantity', 'checkout_time', 'is_prescription']
    list_filter = ['is_prescription', 'checkout_time']
    search_fields = ['astronaut__name', 'medication__name']
    date_hierarchy = 'checkout_time'

@admin.register(InventoryLog)
class InventoryLogAdmin(admin.ModelAdmin):
    list_display = ['medication', 'log_type', 'quantity_change', 'new_quantity', 'timestamp']
    list_filter = ['log_type', 'timestamp']
    search_fields = ['medication__name']
    date_hierarchy = 'timestamp'

@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    list_display = ['event_type', 'astronaut', 'timestamp', 'ip_address']
    list_filter = ['event_type', 'timestamp']
    search_fields = ['astronaut__name', 'description']
    date_hierarchy = 'timestamp'

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