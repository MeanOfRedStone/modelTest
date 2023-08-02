from django.db import models

# Create your models here.
class ModelResult(models.Model):
    story = models.CharField(max_length=200, null=False)
    prediction = models.CharField(max_length=4, null=False)

    class Meta:
        ordering = ['id']