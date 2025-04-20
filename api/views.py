from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from api.serializers import PredictionsSerializerOutput, PredictionsSerializerInput
from api.tasks import get_model_predictions
from api.utils import is_locked


class PredictionsAPIView(APIView):
    def post(self, request):
        input_serializer = PredictionsSerializerInput(data=request.data)
        input_serializer.is_valid(raise_exception=True)

        key = f"{settings.PREDICTIONS_CACHE_KEY}-{input_serializer.data['steps']}"

        cached_predictions = cache.get(key)
        print(key)
        is_processing = is_locked(key)

        if cached_predictions is None:
            if not is_processing:
                get_model_predictions.delay(steps=input_serializer.data['steps'])
            return Response({'status': 'No predictions available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        serializer = PredictionsSerializerOutput(data=cached_predictions, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)

