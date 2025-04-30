from django.urls import path

from api.views import PredictionsAPIView, UpdateSignalsAPIView, HistoryListAPIView

urlpatterns = [
    path('predictions', PredictionsAPIView.as_view(), name='predictions'),
    path('update-signals', UpdateSignalsAPIView.as_view(), name='update_signals'),
    path('history', HistoryListAPIView.as_view(), name='history'),
]