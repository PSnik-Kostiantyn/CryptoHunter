from rest_framework import serializers


class PredictionsSerializer(serializers.Serializer):
    timestamp = serializers.IntegerField()
    close = serializers.FloatField()