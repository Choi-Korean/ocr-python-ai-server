from django.apps import AppConfig

from text_recognition.text_recognition import Text_Recognition


class ControllerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "controller"
    tr = Text_Recognition()
