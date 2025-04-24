from django.urls import path

from api.views import PredictionsAPIView, UpdateSignalsAPIView
urlpatterns = [
    path('predictions', PredictionsAPIView.as_view(), name='predictions'),
    path('update-signals', UpdateSignalsAPIView.as_view(), name='update_signals')
]