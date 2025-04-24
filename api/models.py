from django.db import models

class Signal(models.Model):
    telegram_id = models.CharField(primary_key=True)
    if_below = models.IntegerField()