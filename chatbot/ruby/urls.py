from django.urls import path
from ruby import views

urlpatterns = [
    path('', views.home, name='home'),
    path('error', views.error, name='error'),
]