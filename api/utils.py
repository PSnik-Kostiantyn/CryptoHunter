from django.core.cache import cache


def is_locked(key):
    return cache.get(key) is not None

def acquire_lock(key,timeout=60):
    return cache.add(key, "1", timeout)

def release_lock(key):
    cache.delete(key)