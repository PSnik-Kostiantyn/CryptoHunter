from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from api.models import Signal
from api.serializers import PredictionsSerializerOutput, PredictionsSerializerInput,DecryptedSignalsSerializer
from api.tasks import get_model_predictions
from api.utils import is_locked


class PredictionsAPIView(APIView):
    def post(self, request):
        input_serializer = PredictionsSerializerInput(data=request.data)
        input_serializer.is_valid(raise_exception=True)

        key = f"{settings.PREDICTIONS_CACHE_KEY}-{input_serializer.data['steps']}"

        cached_predictions = cache.get(key)
        is_processing = is_locked(key)

        if cached_predictions is None:
            if not is_processing:
                get_model_predictions.delay(steps=input_serializer.data['steps'])
            return Response({'status': 'No predictions available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        serializer = PredictionsSerializerOutput(data=cached_predictions, many=True)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)

class UpdateSignalsAPIView(APIView):
    def post(self, request):
        serializer = DecryptedSignalsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        existing_telegram_ids = set(Signal.objects.values_list('telegram_id', flat=True))

        create_list = []
        update_list = []

        for signal in serializer.data['signals']:
            telegram_id = signal.get("telegram_id")
            if_below = signal.get("if_below")

            if telegram_id in existing_telegram_ids:
                signal_obj = Signal(telegram_id=telegram_id, if_below=if_below)
                update_list.append(signal_obj)
            else:
                create_list.append(Signal(telegram_id=telegram_id, if_below=if_below))

        if create_list:
            Signal.objects.bulk_create(create_list)

        if update_list:
            Signal.objects.bulk_update(update_list, ['if_below'])

        return Response({'status': 'OK'})