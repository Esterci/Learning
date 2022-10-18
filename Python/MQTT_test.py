# Import standard python modules.
import sys
import time
from Adafruit_IO import MQTTClient
import psutil
import os
import numpy as np

def move(atual):
    time.sleep(1)
    return atual + 0.5


# Set to your Adafruit IO key.
ADAFRUIT_IO_KEY = 'aio_gTQP308mmzqtedc1ruMNq7fH0850'
# Set to your Adafruit IO username.
ADAFRUIT_IO_USERNAME = 'thiagoesterci'

def connected(client):
    print('{}:Connected to Adafruit IO!  Listening for changes...'.format(time.asctime( time.localtime(time.time()) )))
    # Subscribe to changes on a feed .
    client.subscribe('atuador')

def disconnected(client):
    # Disconnected function will be called when the client disconnects.
    print('{}:Disconnected from Adafruit IO!'.format(time.asctime( time.localtime(time.time()) )))
    sys.exit(1)
 




def message(conector,feed_id, payload):
    # Message function will be called when a subscribed feed has a new value.
    # The feed_id parameter identifies the feed, and the payload parameter has
    # the new value.
    print('{}:Feed {} received new value: {}'.format(time.asctime( time.localtime(time.time()) ),feed_id, payload))
    # if(feed_id=='atuador'):
    #  if(payload=='ON'):
    #     print("atuador is turned ON")
    #  elif(payload=='OFF'):
    #     print("atuador is turned OFF")
    #     os.system("shutdown now -h")

# Create an MQTT client instance.
client = MQTTClient(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)
# Setup the callback functions defined above.
client.on_connect    = connected
client.on_disconnect = disconnected
client.on_message    = message
# Connect to the Adafruit IO server.
client.connect()
# Now the program needs to use a client loop function to ensure messages are
# sent and received.  There are a few options for driving the message loop,
# depending on what your program needs to do.

# The first option is to run a thread in the background so you can continue
# doing things in your program.
client.loop_background()
# Now send new values every 10 seconds.
tim=10
tempo = 0
tempo_angulo = 23
velo_vento = 0
seguranca = False
seguranca_tag = 0
angulo_atual = -60
motor_speed = 4
atuador = 0

client.publish('atuador', atuador)
client.publish('seguranca', seguranca_tag)
time.sleep(5)
print('{}:Publishing a new message every ' ,tim,'seconds (press Ctrl-C to quit)...'.format(time.asctime( time.localtime(time.time()) )))
while True:
# gives a single float value

    if velo_vento >= 20:
        seguranca = True
        seguranca_tag = 1
        client.publish('seguranca', seguranca_tag)
    
    angulo_alvo = np.around(np.sin((tempo_angulo/24 * 2* np.pi)) * 60,0)

    if seguranca:
        if angulo_atual < 0:
            angulo_atual += motor_speed 
            atuador = 1

        elif angulo_atual > 0:
            angulo_atual -= motor_speed
            atuador = 1
        
        else:
            atuador = 0
    
    else:
        if (angulo_alvo - angulo_atual) > 0:
            angulo_atual += motor_speed
            atuador = 1

        elif (angulo_alvo - angulo_atual) < 0:
            angulo_atual -= motor_speed
            atuador = 1

        else:
            atuador = 0

    if (tempo % 10) == 0:

        velo_vento += 2
        tempo = 0
        tempo_angulo += 2

        client.publish('angulo', angulo_atual)
        client.publish('angulo_alvo', angulo_alvo)
        client.publish('vento', velo_vento)

        print('{}: angulo value {} {}'.format(time.asctime( time.localtime(time.time()) ),angulo_atual,"°"))
        print('{}: angulo_alvo value {}'.format(time.asctime( time.localtime(time.time()) ),angulo_alvo))
        print('{}: seguranca value {}'.format(time.asctime( time.localtime(time.time()) ),seguranca_tag,"C °"))
        print('{}: vento value {} {}'.format(time.asctime( time.localtime(time.time()) ),velo_vento,"m/s"))
        print("-"*20)
        print("")

            

    print('{}: angulo value {} {}'.format(time.asctime( time.localtime(time.time()) ),angulo_atual,"°"))
    print('{}: angulo_alvo value {}'.format(time.asctime( time.localtime(time.time()) ),angulo_alvo))
    print('{}: tempo_angulo value {}'.format(time.asctime( time.localtime(time.time()) ),tempo_angulo))
    print('{}: seguranca value {}'.format(time.asctime( time.localtime(time.time()) ),seguranca_tag,"C °"))
    print('{}: vento value {} {}'.format(time.asctime( time.localtime(time.time()) ),velo_vento,"m/s"))
    print("-"*20)
    print("")

    time.sleep(1)
    tempo += 2

