from django.urls import path
from .views import TrainingModel

urlpatterns = [
    path('TrainingModel/', TrainingModel, name='TrainingModel')
]
