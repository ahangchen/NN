from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.train, name='index'),
    url(r'^train$', views.train, name='train'),
    url(r'^hyper$', views.hyper, name='hyper'),
]