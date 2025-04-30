from django.db import models

class Signal(models.Model):
    telegram_id = models.CharField(primary_key=True)
    if_below = models.IntegerField()

class History(models.Model):
    timestamp = models.IntegerField()
    predicted = models.FloatField()
    real = models.FloatField(null=True)
