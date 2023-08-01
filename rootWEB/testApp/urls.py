from django.urls import path
from testApp import views

urlpatterns = [
    path('index/', views.index),
    path('transmit/', views.transmit),
    # path('transmit_pos/', views.transmit_pos),
    path('prediction/', views.prediction_list, name='api'),
    path('prediction/<int:pk>/', views.prediction_detail),
]