# Importing required Libraries
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import models, layers
import os
from PIL import Image
import cv2
import numpy as np

# Initializing app
app = Flask(__name__)

# Loading model
from tensorflow.keras.models import load_model
model=load_model("project.h5")

# Creating first end point to render index.html
@app.route('/')
def index():
	return render_template("index.html")

# Function to return equivalence text of digits 0 and 1
def names(number):
    if number==0:
        return 'It\'s a tumor'
    elif number==1:
        return 'It\'s not a tumor'
    else :
        return 'Invalid Image'
# Second end point to render prediction.html
@app.route("/prediction", methods=["POST"])
def prediction():
	
	img = request.files['img'] # Assign the file given by user into img variable

	img_path = "static/pics/" + img.filename #Defined a path to	save the file

	img.save(img_path) # Saved the file

	extension = os.path.splitext(img_path)[1] # stored the extension of file into extension variable
	
	# Making app to only accept image file. If file other than image return Invalid file
	while extension not in ['.jpg','.jpeg','.png','.JPG','.JPEG','.PNG','.tif','.TIF','.gif','.GIF']:
		return render_template("prediction.html",data='Invalid File Format!! '+ extension + ' is not supported. Please choose an appropriate image file.')
	
	# If imputted file is an image, proceed
	else:
		image = Image.open(img_path) #Opening the Image into image variable
		
		image = np.array(image.resize((64,64))) # Resizing the image and converting it to an array
		
		# Checking to see if image has channels or not. No channel means grayscale image. 
		if len(image.shape)==2: # Shape 2 means only (height,width) no channels
			image = np.expand_dims(image, axis=-1) # Adding a channel to the image. Adds channel=1. Shape becomes (height,width,channels)
			image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) # Our model only accepts rgb image so changed the channels to 3 i.e.RGB
		
		if image.shape[2]==4: # Shape 2 means only (height,width) no channels
			image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR) # Our model only accepts rgb image so changed the channels to 3 i.e.RGB
		
		image = np.expand_dims(image,axis=0) #Adding Batch size = 1 to our image since our model takes 4 input parameters (Batch_size,height,width,channels)
		
		res = (model.predict_on_batch(image)) # Feeding the image to our model
		
		classification = np.where(res == np.amax(res))[1][0] # Model gives 2 output. So assigning either 0 or 1 correspoinding to the values into classification variable
		
		a = names(classification) # Message to be displayed.

		return render_template("prediction.html", data=a, user_image=img_path) #Rendering prediction.html and passing message and image to be displayed	

# If this is our main file run it.
if __name__ == "__main__":
	app.run(debug=True)
