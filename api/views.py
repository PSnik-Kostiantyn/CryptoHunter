from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serializers import PredictionsSerializer
from api.tasks import get_model_predictions
from api.utils import is_locked


class PredictionsAPIView(APIView):
    def get(self, request):
        cached_predictions = cache.get(settings.PREDICTIONS_CACHE_KEY)
        is_processing = is_locked(settings.PREDICTIONS_CACHE_KEY)

        if cached_predictions is None:
            if not is_processing:
                get_model_predictions.delay(steps=24)
            return Response({'status': 'No predictions available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        serializer = PredictionsSerializer(data=cached_predictions, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)

