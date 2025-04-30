import subprocess

import pandas as pd
import requests
from celery import shared_task
from django.conf import settings
from django.core.cache import cache
from django.core.paginator import Paginator

from TransformerModel.BTCPriceForecaster import BTCPriceForecaster
from api.models import Signal, History
from api.serializers import EncryptedSignalsSerializer, SignalSerializer
from api.utils import acquire_lock, release_lock
from model_training.dataset_autoupdate import update_btc_dataset


@shared_task
def get_model_predictions(steps):
    key = f"{settings.PREDICTIONS_CACHE_KEY}-{steps}"

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

@shared_task
def dataset_update():
    try:
        subprocess.run(["poetry", "run", "scrapy", "crawl", "news", "-a", f"dataset_path={settings.NEWS_PATH}"])

        forecaster = BTCPriceForecaster()
        forecaster.load_and_prepare_data(data_path=settings.DATASET_PATH)
        forecaster.build_or_load_model(model_path=settings.MODEL_PATH)

        timestamp_from = update_btc_dataset(dataset_path=settings.DATASET_PATH, news_path=settings.NEWS_PATH)

        forecaster.train_model_on_new_data(timestamp_from, csv_path=settings.DATASET_PATH)

        future = forecaster.forecast(steps=24)
        # Update old predictions into history
        df = pd.read_csv(settings.DATASET_PATH)
        history_records = History.objects.filter(real__isnull=True)

        for record in history_records:
            real_price = df.loc[df['Close time'] == record.timestamp]
            if not real_price.empty:
                record.real = real_price.iloc[0]['Close']
                record.save()
        # Store new predictions into history
        histories = [History(timestamp=prediction['timestamp'], predicted=prediction['close']) for prediction in future]
        History.objects.bulk_create(histories)
        # Notify users
        min_price = min(list(map(lambda item: item.get('close'), future)))
        signals = Signal.objects.filter(if_below__lt=min_price).order_by("telegram_id")
        paginator = Paginator(signals, 100)
        for page_num in paginator.page_range:
            page = paginator.page(page_num)
            signals_data = SignalSerializer(page.object_list, many=True).data
            serializer = EncryptedSignalsSerializer(data={"signals": signals_data})
            if serializer.is_valid():
                requests.post(settings.REPORTER_SERVER_URL, json=serializer.data)

    except Exception as e:
        print(e)

