from django.urls import path

from api.views import PredictionsAPIView

urlpatterns = [
    path('predictions', PredictionsAPIView.as_view(), name='predictions'),
]