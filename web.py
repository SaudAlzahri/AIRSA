import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import requests
from io import BytesIO
import torch


# set full screen width
st.set_page_config(layout="wide", page_title="AIRSA: Road Safety Assessment Tool", page_icon="🚦",menu_items={'About': 'https://github.com/SaudAlzahri/AIRSA'})
st.header("AIRSA — Artificial Intelligence Road Safety Analysis")

original_title = '<p style="font-family:Courier; color:Cyan; font-size: 14px;">AIRSA\'s goal is to provide statistics on road safety using AI recognition. The AI model recognizes key factors of road safety (such as traffic light presence and stop sign presence) that are used in the safety formula. Other outputs such as road width, lane count, and individual lane width are factors of the safety formula.</p>'
st.markdown(original_title, unsafe_allow_html=True)


st.write(
    ":rocket:"
)

orinal_titl = '<p style="font-family:Courier; color:Gold; font-size: 24px;">RESULTS</p>'
st.markdown(str(orinal_titl), unsafe_allow_html=True)
uouo = '<p style="font-family:Courier; color:Gold; font-size: 9px;">(Click the arrow in the top left corner for inputs. The section is labeled \'TRY IT OUT!\')</p>'
st.markdown(str(uouo), unsafe_allow_html=True)


original_titl = '<p style="font-family:Courier; color:Cyan; font-size: 30px;">TRY IT OUT!</p>'
st.sidebar.markdown(str(original_titl), unsafe_allow_html=True)

#img_url input
oriinal_titl = '<p style="font-family:Courier; color:Gold; font-size: 17px;">IMAGE URL</p>'
st.sidebar.markdown(str(oriinal_titl), unsafe_allow_html=True)
img_url = st.sidebar.text_input('', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsP7xEOzp5Ii4nV4STAR9HjnVJqPfJLz1rSwqN-qsp&s')
img_url_invalid = False
if img_url == "":
	img_url_invalid = True
	riginal_titl = '<p style="font-family:Courier; color:Red; font-size: 17px;">You left the IMAGE URL box empty.</p>'
	st.sidebar.markdown(str(riginal_titl), unsafe_allow_html=True)
else:
	pass

st.sidebar.markdown("""---""")
#roadwidth input
try:
	oriinl_titl = '<p style="font-family:Courier; color:Gold; font-size: 17px;">ROAD WIDTH (METERS)</p>'
	st.sidebar.markdown(str(oriinl_titl), unsafe_allow_html=True)
	roadwidth = int(st.sidebar.text_input('', '14'))
except ValueError:
	riinal_titl = '<p style="font-family:Courier; color:Red; font-size: 17px;">You left the ROAD WIDTH box empty.</p>'
	st.sidebar.markdown(str(riinal_titl), unsafe_allow_html=True)
st.sidebar.markdown("""---""")
#lanecount inout
try:
	orinl_titl = '<p style="font-family:Courier; color:Gold; font-size: 17px;">LANE COUNT</p>'
	st.sidebar.markdown(str(orinl_titl), unsafe_allow_html=True)
	lanecount = int(st.sidebar.text_input('', '5'))
except ValueError:
	riinal_titl = '<p style="font-family:Courier; color:Red; font-size: 17px;">You left the LANE COUNT box empty.</p>'
	st.sidebar.markdown(str(riinal_titl), unsafe_allow_html=True)
st.sidebar.header(":gear:")




######################### _-_-_

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

im = 'HiddenStop.2.jpg'

# Inference
results = model(im)
r_img = results.render() # returns a list with the images as np.array
img_with_boxes = r_img[0] # image with boxes as np.array


##########################



col1, col2 = st.columns(2)


try:
	response = requests.get(img_url)
	my_image = Image.open(BytesIO(response.content))
# alpha_matting = st.sidebar.checkbox("Include alpha matting (can sometimes improve removal)", value=False)
# if alpha_matting:
#     alpha_matting_background_threshold = st.sidebar.number_input(
#         "Alpha matting background", value=10, min_value=0, max_value=2000, step=1
#     )
#     alpha_matting_foreground_threshold = st.sidebar.number_input(
#         "Alpha matting foreground", value=240, min_value=0, max_value=500, step=5
#     )






#################################




# if st.button('Train model'):
#     with st.spinner("Training ongoing"):
#         clf, confusion_matrix = train_rf(df, cols_to_train)
#         st.balloons()
#         st.write(confusion_matrix)







# YOLO VERSION 5 GETS THE IMAGE AND RETURNS IT IN MODEL AS A STRING
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n - yolov5x6 official model
except AttributeError:
	pass
######################### _-_-_


# Image

# Inference
results = model(img_url)
r_img = results.render() # returns a list with the images as np.array
img_with_boxes = r_img[0] # image with boxes as np.array


##########################


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
try:
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
except NameError:
	pass

#INDIVIDUAL LANE WIDTH COUNTS FOR 45%
try:
	lanewidth = roadwidth / int(lanecount)
	if lanewidth <= 2:
	    score=score +8
	elif lanewidth <= 2.3:
	    score=score+15
	elif lanewidth <= 2.65:
	    score=score+27
	elif lanewidth <= 3:
	    score =score+37
	else:
	    score= score+45
except NameError:
	pass


#LANE COUNT COUNTS FOR 20%
try:
	if lanecount <= 2:
	    score =score+1
	elif lanecount <= 4:
	    score = score + 10
	elif score <= 5:
	    score =score +15
	else:
	    score = score + 20
except NameError:
	pass


# #PRINTING OUTPUT FINALLY!
# from colorama import Fore
# from colorama import Style
# #THIS MESSAGE IS SELF EXPLANATORY, IS TRIGGERING BY EXTREMELY LOW SAFETY SCORE
# if score <= 40:
#     st.write('\033[1m' + f"{Fore.RED}WARNING: BLACKSPOT - HIGH POTENTIAL CAR CRASH SITE!{Style.RESET_ALL}" + '\033[0m')

# #THIS IS THE PROGRAM'S FINAL STATISTICS ON THE ROAD
# st.write('STATISTICS')
# st.write("Safety Score: " + str(score) + "%")
# st.write("Width of lane:", str(roadwidth / int(lanecount)) + " meters")
# st.write("Approx. cars per lane:", str(carcount/lanecount))
# st.write("Presence of traffic light:", trafficlightpresence)
# st.write("Presence of stop sign:", stopsignpresence)
# st.write("Total cars:", str(carcount))



col1.write("INSERTED IMAGE")
col1.image(img_with_boxes)
#     if alpha_matting:
#         fixed = remove(
#             image,
#             alpha_matting=alpha_matting,
#             alpha_matting_background_threshold=alpha_matting_background_threshold,
#             alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
#         )
#     else:
if score <= 40:
	st.write("WARNING: BLACKSPOT - HIGH POTENTIAL CAR CRASH SITE")
else:
	pass

try:
	data = {'Safety Score': str(score),
		'Lane Width': str(roadwidth / int(lanecount)) + "m",
		'Approx. cars per lane': str(carcount/lanecount),
		'Traffic Light Presence':trafficlightpresence,
		'Stop Sign Presence': stopsignpresence,
		'Car Count': str(carcount)}
	df = pd.DataFrame(data,index=[0]).transpose()
	
	
	with col2:
		st.write('STATISTICS')

		st.dataframe(df,use_container_width=True)
		st.sidebar.markdown("\n")
except NameError:
	with col2:
		st.write('STATISTICS')
		riina_titl = '<p style="font-family:Courier; color:Red; font-size: 17px;">The dataframe cannot be loaded because you left an input box empty.</p>'
		st.markdown(str(riina_titl), unsafe_allow_html=True)
		st.sidebar.markdown("\n")


st.write("In the future AIRSA will be able to provide more advanced statistics than already, hopefully supporting pedestrian safety by recognizing sidewalk presence.")

st.write(
	":satellite:"
)

original_tite = '<p style="font-family:Courier; color:Cyan; font-size: 33px;">STATISTICS ON ROAD SAFETY IN RIYADH, SAUDI ARABIA - analyzed by AIRSA</p>'
st.markdown(original_tite, unsafe_allow_html=True)


st.write("Description of study: Our team took a total of 50 pictures of Riyadh's roads. Using AIRSA to analyze the results, we created the heat map below; the closer to green the higher the score (safe), and the closer to red the lower the score (unsafe).")

coli, coly = st.columns(2)

with coli:
	
	heat = '<p style="font-family:Courier; color:Gold; font-size: 17px;">HEAT MAP</p>'
	st.markdown(str(heat), unsafe_allow_html=True)
	
	response_study = requests.get('https://i.imgur.com/wVVVrDC.png')
	my_study_image = Image.open(BytesIO(response_study.content))
	st.image(my_study_image)


	
datanew = {
	'Average Score': "60.68",
	'Average Lane Width': "5.04 meters",
	'Average Traffic Light Presence': "18%",
	'Average Stop Sign Presence': "12%"
}



rf = pd.DataFrame(datanew,index=[0]).transpose()
with coly:
	average = '<p style="font-family:Courier; color:Gold; font-size: 17px;">AVERAGES</p>'
	st.markdown(str(average), unsafe_allow_html=True)
	
	st.dataframe(rf,use_container_width=True)

	

url = "https://github.com/SaudAlzahri/AIRSA"
st.markdown("[:gear:PROJECT CODE:gear:](%s)" % url)

st.write(
    ":flying_saucer:"
)
original__title = '<p style="font-family:Courier; color:Gray; font-size: 20px;">THE AIRSA TEAM</p>'
st.markdown(original__title, unsafe_allow_html=True)

originaltitle = '<p style="font-family:Courier; color:Gray; font-size: 15px;">Saud Alzahri, Ahmad Almalik, Jude Alhazmi</p>'
st.markdown(originaltitle, unsafe_allow_html=True)
st.write("""
""")
orginaltitle = '<p style="font-family:Courier; color:Gray; font-size: 15px;">CREDITS:</p>'
st.markdown(orginaltitle, unsafe_allow_html=True)
oginaltitle = '<p style="font-family:Courier; color:Gray; font-size: 15px;">Tons of thanks to Mr. Panos and Naif Alzahri for  their continuous support and advice.</p>'
st.markdown(oginaltitle, unsafe_allow_html=True)

st.write("""
""")

ognaltitle = '<p style="font-family:Courier; color:Gray; font-size: 15px;">Credit to YOLO (v5)</p>'
st.markdown(ognaltitle, unsafe_allow_html=True)

ognltitle = '<p style="font-family:Courier; color:Gray; font-size: 15px;">Yᵒᵘ Oᶰˡʸ Lᵒᵒᵏ Oᶰᶜᵉ</p>'
st.markdown(ognltitle, unsafe_allow_html=True)

################################### _-_-_

