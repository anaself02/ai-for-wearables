from django.contrib import admin

from .models import HistoricalMetric


@admin.register(HistoricalMetric)
class HistoricalMetricAdmin(admin.ModelAdmin):
    list_display = ("user", "metric_type", "date", "value", "unit", "source")
    list_filter = ("metric_type", "source")
    search_fields = ("user__username",)
    date_hierarchy = "date"
    readonly_fields = ("imported_at",)
