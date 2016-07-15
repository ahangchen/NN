from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.train, name='index'),
    url(r'^$', views.hyper, name='index'),
]