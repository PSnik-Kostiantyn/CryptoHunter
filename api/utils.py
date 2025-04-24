import base64

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from django.conf import settings
from django.core.cache import cache
from openai import APIError
from rest_framework.exceptions import ValidationError

LOCK_PREFIX='lock:'

def is_locked(key):
    return cache.get(LOCK_PREFIX+key) is not None

def acquire_lock(key,timeout=60):
    return cache.add(LOCK_PREFIX+key, "1", timeout)

def release_lock(key):
    cache.delete(LOCK_PREFIX+key)

def encrypt_message(message):
    cipher = AES.new(settings.SECRET_KEY.encode(), AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return iv, ct

def decrypt_message(iv_base64, ciphertext_base64):
    try:
        iv = base64.b64decode(iv_base64)
        ciphertext = base64.b64decode(ciphertext_base64)
        cipher = AES.new(settings.SECRET_KEY.encode(), AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_data.decode('utf-8')
    except ValueError as e:
        raise ValidationError(e)