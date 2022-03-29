import keyboard #klavye erisimi
import uuid # ekran kaydi
import time
from PIL import Image
from mss import mss #pixselli alani kesip frame e donusturuyor

"""
http://www.trex-game.skipser.com/
"""

mon = {"top":360, "left":505, "width":250, "height":100} 
sct = mss()

i =0

def record_screen(record_id, key):
    global i
    
    i+=1
    print("{} {} ".format(key,i)) #key hangi tusa bastigimiz, i kac kere bastigimiz
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key,record_id,i))
    
is_exit = False
def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc",exit) # exis fonksiyonunu cagiracak esc ye basinca

record_id = uuid.uuid4()

while True:
    
    if is_exit == True:
        break
    try:
        if(keyboard.is_pressed(keyboard.KEY_UP)):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif(keyboard.is_pressed(keyboard.KEY_DOWN)):
            record_screen(record_id,"down")
            time.sleep(0.1)
        elif(keyboard.is_pressed("right")):
            record_screen(record_id,"right")
            time.sleep(0.1)
    except RuntimeError: continue





























    

