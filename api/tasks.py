from celery import shared_task
from django.conf import settings
from django.core.cache import cache

from TransformerModel.BTCPriceForecaster import BTCPriceForecaster
from api.utils import acquire_lock, release_lock


@shared_task
def get_model_predictions(steps):
    key = f"{settings.PREDICTIONS_CACHE_KEY}-{steps}"
    print(key)

    acquire_lock(key, timeout=60)
    try:
        forecaster = BTCPriceForecaster()
        forecaster.load_and_prepare_data(data_path=settings.DATASET_PATH)
        forecaster.build_or_load_model(model_path=settings.MODEL_PATH)
        predictions = forecaster.forecast(steps)
        cache.set(key, predictions, timeout=settings.PREDICTIONS_CACHE_TIMEOUT)
    except Exception as e:
        print(e)
        cache.delete(key)
    finally:
        release_lock(key)
