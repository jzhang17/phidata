from django import forms
from .models import Research

class ResearchForm(forms.ModelForm):
    class Meta:
        model = Research
        fields = ['query']
from django import forms
from .models import Research

class ResearchForm(forms.ModelForm):
    class Meta:
        model = Research
        fields = ['query']
