# Importing necessary libraries
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# Loading the pre-trained model:
model = load_model('Model.h5')
import numpy as np
import os
# Creating a Flask web application:
app = Flask(__name__)
# Defining a route for the home page:
# This route renders the 'index.html' template when the user visits the home page.
@app.route('/')
def index():
    return render_template('index.html')
# Defining a route for handling file uploads:
# This route is triggered when the user uploads an image via a POST request.
@app.route('/upload', methods=['POST'])
def upload():
  # Handling file upload and saving the image:
  # This code retrieves the uploaded file, secures the filename,
  # saves the file in the 'static' folder, and constructs the path to the saved file.
    file = request.files['file']

    filename = secure_filename(file.filename)
    print(filename)
    file.save(os.path.join('static', filename))
    path="static/"+filename
    print(path)
    from tensorflow.keras.preprocessing import image


# testing the model
# Testing the model on the uploaded image:
# This section preprocesses the uploaded image and uses the pre-trained model to make predictions.
    test_image = image.load_img(path, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(x= test_image)
    print(result)
    # Determining the predicted class label:
    # The code checks the index of the highest probability in the prediction results and assigns a corresponding class label.
    if np.argmax(result)  == 0:
      prediction = 'Bread'
    elif np.argmax(result)  == 1:
      prediction = 'Dairyproduct'
    elif np.argmax(result)  == 2:
      prediction ='Dessert'
    elif np.argmax(result)  == 3:
      prediction ='Egg'
    elif np.argmax(result)  == 4:
      prediction ='Friedfood'
    elif np.argmax(result)  == 5:
      prediction ='Meat'
    elif np.argmax(result)  == 6:
      prediction ='Noodles-Pasta'
    elif np.argmax(result)  == 7:
      prediction ='Rice'
    elif np.argmax(result)  == 8:
      prediction ='Seafood'
    elif np.argmax(result)  == 9:
      prediction ='Soup'
    elif np.argmax(result)  == 10:
      prediction ='Vegetable-Fruit'

    print(prediction,"dddsaszdss")
# Rendering the result in the HTML template:
# The predicted class label is passed to the 'index.html' template.
    return render_template('index.html',Data=prediction)

# Running the Flask application:
# This ensures that the Flask application runs when the script is executed
if __name__ == '__main__':
    app.run()

# In summary, this Flask application allows users to upload an image,
# processes the image using a pre-trained model, and displays the prediction on the web page.
# The result is rendered using the 'index.html' template.