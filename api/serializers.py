from rest_framework import serializers


class PredictionsSerializerOutput(serializers.Serializer):
    timestamp = serializers.IntegerField()
    close = serializers.FloatField()

class PredictionsSerializerInput(serializers.Serializer):
    steps = serializers.IntegerField(min_value=1, max_value=120, default=6)