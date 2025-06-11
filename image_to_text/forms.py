# image_to_text/forms.py
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class TextToImageForm(forms.Form):
    text = forms.CharField(label='Enter your text', max_length=100)

