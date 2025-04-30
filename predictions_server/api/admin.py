from django.contrib import admin

from api.models import Signal

@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ('telegram_id', 'if_below')
