from celery import shared_task
from django.conf import settings
from django.core.cache import cache

from TransformerModel.BTCPriceForecaster import BTCPriceForecaster
from api.utils import acquire_lock, release_lock


@shared_task
def get_model_predictions(steps):
    acquire_lock(settings.PREDICTIONS_CACHE_KEY, timeout=60)
    try:
        forecaster = BTCPriceForecaster()
        forecaster.load_and_prepare_data(data_path=settings.DATASET_PATH)
        forecaster.build_or_load_model(model_path=settings.MODEL_PATH)
        predictions = forecaster.forecast(steps)
        cache.set(settings.PREDICTIONS_CACHE_KEY, predictions, timeout=settings.PREDICTIONS_CACHE_TIMEOUT)
    except Exception as e:
        print(e)
        cache.delete(settings.PREDICTIONS_CACHE_KEY)
    finally:
        release_lock(settings.PREDICTIONS_CACHE_KEY)
