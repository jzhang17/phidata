from django.db import models

class Research(models.Model):
    query = models.CharField(max_length=255)
    results = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.query
from django.db import models

class Research(models.Model):
    query = models.CharField(max_length=255)
    results = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.query
