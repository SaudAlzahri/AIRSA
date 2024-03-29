import torch

img_url = input('\033[1m' + "Insert image address: " + '\033[0m')
roadwidth = int((input('\033[1m' + "How wide is the road? " + '\033[0m')).replace(" meters", "").replace("m", "").replace(" m", ""))
lanecount = int(input('\033[1m' + "How many lanes are there? " + '\033[0m'))




# YOLO VERSION 5 GETS THE IMAGE AND RETURNS IT IN MODEL AS A STRING
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n - yolov5x6 official model

# Inference
results = model(img_url)



# YOLO'S STRING NEEDS FILTERING OUT THE TRASH WORDS THAT ARE NOT GOING BE CONVERTED TO USABLE VALUES
# THE 'GOODNESS' OF THESE FILTERING TECHNIQUES COULD'VE BEEN BETTER BUT IT STILL WORKS
# WHAT I MEAN IS THE FILTER COULD'VE BEEN MUCH MORE EFFICIENT IN THE CONCEPT OF MINIMAL LINES, BUT ITS OKAY!

waw, removethispiece__ = str(results).split("Speed: ")
waw = waw.replace("image 1/1: ", "")
waw = (waw[7:]).upper()

import re
def subit(m):
    stuff, word = m.groups()
    return ("_" * len(stuff)) + word
waw = (re.sub(r'(.+?)(CAR|STOP SIGN|TRAFFIC LIGHT|BUS|TRUCK|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|118|19|20|21|22|23|24|25|$)', subit, waw))
waw = waw.replace("__", "  ").replace("_", " ").replace("  ", " ")
waw = waw.replace("  1   ", "  ").replace("  2   ", "  ").replace("  3   ", "  ").replace("  4   ", "  ").replace("  5   ", "  ").replace("  6   ", "  ").replace("  7   ", "  ").replace("  8   ", "  ").replace("  9   ", "  ").replace("  10   ", "  ").replace("  11   ", "  ").replace("  12   ", "  ").replace("  13   ", "  ").replace("  14   ", "  ").replace("  15   ", "  ").replace("  16   ", "  ").replace("  17   ", "  ").replace("  18   ", "  ").replace("  19   ", "  ").replace("  20   ", "  ").replace("  21   ", "  ").replace("  22   ", "  ").replace("  23   ", "  ").replace("  24   ", "  ").replace("  25   ", "  ").replace("  26   ", "  ").replace("  27   ", "  ").replace("  28   ", "  ").replace("  29   ", "  ").replace("  30   ", "  ").replace("  31   ", "  ").replace("    ", " ")


## LINES 48 TO 90 CREATE VALUES BASED OFF OF THE STRING OUTPUTTED BY YOLO..  (THE STRING THAT WE FILTERED THROUGH ABOVE)
#         --EXAMPLE:--
## YOLO GIVES A STRING '3 CAR 4 BUS 1 TRUCK'
## THEN BECOMES...
##### CARCOUNT = 3
##### TRAFFICLIGHTCOUNT = 4
##### STOPSIGNCOUNT = 1

####################
#COUNTS STOP SIGNS
stopsignindex = waw.find("STOP SIGN")
if stopsignindex == -1:
    stopsigncount = 0
else:
    stopsigncount = waw[((int(stopsignindex))-2):((int(stopsignindex))-1)]

####################
#COUNTS TRAFFIC LIGHTS
trafficlightindex = waw.find("TRAFFIC LIGHT")
if trafficlightindex == -1:
    trafficlightcount = 0
else:
    trafficlightcount = int(waw[((int(trafficlightindex))-2):((int(trafficlightindex))-1)])
    
########################
#COUNTS CAR + TRUCK + BUS
truckindex = waw.find("TRUCK")
if truckindex == -1:
    truckcount = 0
else:
    truckcount = waw[((int(truckindex))-2):(int(truckindex)-1)]

busindex = waw.find("BUS")
if busindex == -1:
    buscount = 0
else:
    buscount = waw[((int(busindex))-2):(int(busindex)-1)]


carindex = waw.find("CAR")
if carindex == -1:
    car_count = 0
else:
    car_count = waw[((int(carindex))-2):((int(carindex))-1)]
carcount = int(car_count) + int(truckcount) + int(buscount)




trafficlightcount = int(trafficlightcount)
truckcount = int(truckcount)
stopsigncount = int(stopsigncount)
carcount = int(carcount)

if trafficlightcount >= 1:
    trafficlightpresence = "TRUE"
else:
    trafficlightpresence = "FALSE"
if stopsigncount >= 1:
    stopsignpresence = "TRUE"
else:
    stopsignpresence = "FALSE"





# THE BELOW IS ANALYZING VALUES THAT WE GOT ABOVE   (FROM YOLO'S STRINGS AND THE MANUAL QUESTIONS)


# THE VALUES WILL BE PUT IN A SAFETY FORMULA THAT SEEMS LOGICAL TO ME
score = 0


#TRAFFIC LIGHT OR STOP SIGN COUNTS FOR 30%
if trafficlightpresence =="TRUE" or stopsignpresence == "TRUE":
    score=score+30
else:
    pass



#ROAD WIDTH COUNTS FOR 5%
if roadwidth <= 6:
    score=score+1
elif roadwidth <= 9:
    score=score+2
elif roadwidth <= 13:
    score =score+3
elif roadwidth <= 22:
    score = score + 4
else:
    score =+ 5

#INDIVIDUAL LANE WIDTH COUNTS FOR 45%
lanewidth = roadwidth / int(lanecount)
if lanewidth <= 2:
    score=score +10
elif lanewidth <= 2.3:
    score=score+15
elif lanewidth <= 2.65:
    score=score+27
elif lanewidth <= 3:
    score =score+37
else:
    score= score+45



#LANE COUNT COUNTS FOR 20%
if lanecount <= 2:
    score =score+5
elif lanecount <= 4:
    score = score + 10
elif score <= 5:
    score =score +15
else:
    score = score + 20



#PRINTING OUTPUT FINALLY!
print("""


""")
from colorama import Fore
from colorama import Style
#THIS MESSAGE IS SELF EXPLANATORY, IS TRIGGERING BY EXTREMELY LOW SAFETY SCORE
if score <= 40:
    print('\033[1m' + f"{Fore.RED}WARNING: BLACKSPOT - HIGH POTENTIAL CAR CRASH SITE!{Style.RESET_ALL}" + '\033[0m')

#THIS IS THE PROGRAM'S FINAL STATISTICS ON THE ROAD
print('\033[1m' + 'STATISTICS' + '\033[0m')
print("Safety Score: " + str(score) + "%")
print("Width of lane:", str(roadwidth / int(lanecount)) + " meters")
print("Approx. cars per lane:", carcount/lanecount)
print("Presence of traffic light:", trafficlightpresence)
print("Presence of stop sign:", stopsignpresence)
print("Total cars:", carcount)


#CREDITS
print("""


Tons of thanks to Mr. Panos for his continuous support and advice.

Credit to YOLO (v5)
Yᵒᵘ Oᶰˡʸ Lᶤᵛᵉ Oᶰᶜᵉ
""")
