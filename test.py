from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved Keras model
model = load_model('Model.h5')
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(x= test_image)
    print(result)
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

    print( prediction)


print(testing_image(r'C:\Users\user\Desktop\Food\evaluation\Dairy product\1.jpg'))

