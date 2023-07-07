from django.urls import path
from testApp import views

urlpatterns = [
    path('index/', views.index),
    path('transmit/', views.transmit),
]