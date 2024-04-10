from django.urls import path
from .views import GenerateTrainData,sse

urlpatterns = [
    path('GenerateTrainData/', GenerateTrainData, name='GenerateTrainData'),
    path('sse/', sse, name='sse')
]
