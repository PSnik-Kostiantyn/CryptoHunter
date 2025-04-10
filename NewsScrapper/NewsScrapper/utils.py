import random
import re
import time
from datetime import datetime, timezone, timedelta
from functools import wraps


def retry_on_exception(times: int = 5, backoff_base=2, backoff_max=60, jitter=True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, response, *args, **kwargs):
            try:
                for result in func(self, response, *args, **kwargs):
                    yield result
            except Exception as e:
                retry_times = response.meta.get('retry_times', 0)

                delay = min(backoff_base ** retry_times, backoff_max)
                if jitter:
                    delay = delay * random.uniform(0.8, 1.2)

                self.logger.warning(f"Error parsing {response.url}: {e}")
                time.sleep(delay)

                if retry_times < times:
                    self.logger.info(f"Retrying {response.url} ({retry_times + 1}/{times})")
                    retry_req = response.request.copy()
                    retry_req.meta['retry_times'] = retry_times + 1
                    retry_req.dont_filter = True
                    yield retry_req
                else:
                    self.logger.error(f"Failed after{times} retries: {response.url}")

        return wrapper

    return decorator


def remove_html_tags(text: str) -> str:
    clean_text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    return clean_text


def get_iso_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)


def get_last_full_hour_utc(safety_margin_minutes=5):
    now = datetime.now(timezone.utc)
    adjusted_time = now - timedelta(minutes=safety_margin_minutes)
    last_full_hour = adjusted_time.replace(minute=0, second=0, microsecond=0)
    return last_full_hour


def get_current_full_hour_utc(safety_margin_minutes=5):
    now = datetime.now(timezone.utc)
    if now.minute < safety_margin_minutes:
        current_hour = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        current_hour = now.replace(minute=0, second=0, microsecond=0)
    return current_hour
