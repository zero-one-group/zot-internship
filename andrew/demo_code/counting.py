import signal, time

def handler(signum, time):
    print("\nI got a SIGINT, but I am not stopping")

signal.signal(signal.SIGINT, handler)
count = 0
while True:
    time.sleep(.1)
    print ("\r{}".format(count), end="")
    count += 1

