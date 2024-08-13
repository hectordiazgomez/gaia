from django.urls import path
from . import views

urlpatterns = [
    path('train-nmt/', views.train_nmt, name='train_nmt'),
    path('translate/', views.translation_endpoint, name='translation_endpoint'),
]
