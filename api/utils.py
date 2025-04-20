from django.core.cache import cache

LOCK_PREFIX='lock:'

def is_locked(key):
    return cache.get(LOCK_PREFIX+key) is not None

def acquire_lock(key,timeout=60):
    return cache.add(LOCK_PREFIX+key, "1", timeout)

def release_lock(key):
    cache.delete(LOCK_PREFIX+key)