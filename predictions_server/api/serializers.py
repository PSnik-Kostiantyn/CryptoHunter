import json

from rest_framework import serializers

from api.models import History
from api.utils import encrypt_message, decrypt_message


class PredictionsSerializerOutput(serializers.Serializer):
    timestamp = serializers.IntegerField()
    close = serializers.FloatField()


class PredictionsSerializerInput(serializers.Serializer):
    steps = serializers.IntegerField(min_value=1, max_value=120, default=6)


class SignalSerializer(serializers.Serializer):
    telegram_id = serializers.CharField()
    if_below = serializers.IntegerField()


class EncryptedSignalsSerializer(serializers.Serializer):
    signals = SignalSerializer(many=True, write_only=True)
    iv = serializers.CharField(read_only=True)
    ct = serializers.CharField(read_only=True)

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        iv, ct = encrypt_message(json.dumps(instance.get('signals')))
        representation['iv'] = iv
        representation['ct'] = ct
        return representation

class DecryptedSignalsSerializer(serializers.Serializer):
    signals = SignalSerializer(many=True, read_only=True)
    iv = serializers.CharField(write_only=True)
    ct = serializers.CharField(write_only=True)

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        iv, ct = instance.get('iv'), instance.get('ct')
        representation['signals'] = list(json.loads(decrypt_message(iv, ct)))
        return representation

class HistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = History
        fields = ('timestamp', 'predicted', 'real')