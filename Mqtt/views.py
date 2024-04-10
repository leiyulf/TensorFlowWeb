from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import random
import time
from paho.mqtt import client as mqtt_client
from django.conf import settings

broker = settings.MQTT_BROKER_URL
port = settings.MQTT_BROKER_PORT
topic = "/python/mqtt"
client_id = f'python-mqtt-{int(time.time())}'

mqtt_client_instance = None

def connectMqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publishMessage(client):
    msg_count = 0
    while True:
        time.sleep(5)
        msg = f"messages: {msg_count}"
        result, _ = client.publish(topic, msg)
        if result == mqtt_client.MQTT_ERR_SUCCESS:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1

@csrf_exempt
@require_POST
def startPublishing(request):
    global mqtt_client_instance
    if mqtt_client_instance is None:
        mqtt_client_instance = connectMqtt()
        mqtt_client_instance.loop_start()
        publishMessage(mqtt_client_instance)
        return JsonResponse({'message': 'MQTT publishing started successfully.'})
    else:
        return JsonResponse({'message': 'MQTT publishing is already started.'})

@csrf_exempt
@require_POST
def stopPublishing(request):
    global mqtt_client_instance
    if mqtt_client_instance is not None:
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()
        mqtt_client_instance = None
        return JsonResponse({'message': 'MQTT publishing stopped successfully.'})
    else:
        return JsonResponse({'message': 'No MQTT publishing is running.'})
