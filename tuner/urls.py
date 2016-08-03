from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.train, name='index'),
    url(r'^fit/$', views.train, name='train'),
    url(r'^hyper/$', views.hyper, name='hyper'),
]