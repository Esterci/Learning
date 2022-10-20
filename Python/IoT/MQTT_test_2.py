# Import standard python modules.
import sys
import time
from Adafruit_IO import MQTTClient
import os


# Set to your Adafruit IO key.
ADAFRUIT_IO_KEY = 'aio_pFRc02jlf1rFCruYFIHoACw26RNH'
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
    if(feed_id=='atuador'):
     if(payload=='1'):
        print("atuador is turned ON")
        client.publish('Door', '1')
     elif(payload=='0'):
        print("atuador is turned OFF")
        os.system("shutdown now -h")

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

time.sleep(5)
print('{}:Publishing a new message every ' ,tim,'seconds (press Ctrl-C to quit)...'.format(time.asctime( time.localtime(time.time()) )))
while True:
# gives a single float value


 

    client.publish('vento', ldr_value)

          
    print('{}: vento value {} {}'.format(time.asctime( time.localtime(time.time()) ),ldr_value,"m/s"))
    print("-"*20)
    print("")

    time.sleep(15)

