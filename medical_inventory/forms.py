from django import forms
from .models import Medication  # Adjust import based on your model location

class MedicationForm(forms.ModelForm):
    class Meta:
        model = Medication
        fields = ['name', 'medication_type', 'dosage', 'current_quantity', 
                  'minimum_quantity', 'container_location', 'expiration_date']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Ibuprofen'}),
            'medication_type': forms.Select(attrs={'class': 'form-control'}),
            'dosage': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 200mg'}),
            'current_quantity': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'minimum_quantity': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'container_location': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Cabinet A-1'}),
            'expiration_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
        }