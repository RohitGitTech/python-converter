# image_to_text/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('generate/', views.generate_image, name='generate_image'),
    path('animate/', views.animate_image, name='animate_image'),


]
