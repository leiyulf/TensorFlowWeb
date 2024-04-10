from django.urls import path
from .views import ModelPrediction

urlpatterns = [
    path('ModelPrediction/', ModelPrediction, name='ModelPrediction')
]
