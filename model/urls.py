from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('home/', views.say_hello),
    path('history/', views.load_history)
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
