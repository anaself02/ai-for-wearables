"""Create the two course-project participants.

Anas collects on a Fitbit Air and his data is loaded from a Google Takeout
export. Rahil collects on an EmotiBit; his export arrives after the
mid-project presentation, so he starts with no data and no parser is claimed
for his format.
"""

from django.db import migrations

PEOPLE = [
    {
        "slug": "anas",
        "display_name": "Anas",
        "device_label": "Fitbit Air",
        "adapter_key": "fitbit-takeout",
        "timezone": "America/Toronto",
        "notes": "Google Takeout heart_rate CSV files.",
    },
    {
        "slug": "rahil",
        "display_name": "Rahil",
        "device_label": "EmotiBit",
        "adapter_key": "emotibit",
        "timezone": "America/Toronto",
        "notes": "Awaiting data: the EmotiBit export is expected after the "
                 "mid-project presentation.",
    },
]


def seed(apps, schema_editor):
    Person = apps.get_model("people", "Person")
    for row in PEOPLE:
        Person.objects.get_or_create(slug=row["slug"], defaults=row)


def unseed(apps, schema_editor):
    Person = apps.get_model("people", "Person")
    Person.objects.filter(slug__in=[row["slug"] for row in PEOPLE],
                          heart_rate_days__isnull=True).delete()


class Migration(migrations.Migration):
    dependencies = [("people", "0001_initial")]
    operations = [migrations.RunPython(seed, unseed)]
