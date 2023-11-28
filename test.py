# import cv2
# from keras.models import load_model
# from PIL import Image
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import normalize

# test_datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# model= load_model('BrainTumorImproved.h5')

# image = cv2.imread("C:\\Users\\Ayesha Nadeem\\OneDrive\\Documents\\semester 5\\AI project\\dataset\\Testing\\glioma_tumor\\image(5).jpg")

# img=Image.fromarray(image)
# img=img.resize((64,64))
# img=np.array(img)
# data = np.expand_dims(img, axis=0)
# #print(img)

# data_batch = np.repeat(data, 16, axis=0)  # Assuming batch_size is 16

# # Apply data augmentation
# data_augmented = test_datagen.flow(data_batch, shuffle=False)
# result = model.predict(data_augmented)
# #result = model.predict(data)
# print(result)

# # predicted_class = np.argmax(result)
# # print("Predicted Class:", predicted_class)




#code 2
import cv2
import numpy as np
from keras.models import load_model
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('BrainTumorImproved.h5')

# Load and preprocess the image
image_path = "C:\\Users\\Ayesha Nadeem\\OneDrive\\Documents\\semester 5\\AI project\\dataset\\Testing\\meningioma_tumor\\image(1).jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Predict the class
result = model.predict(image)
predicted_class = np.argmax(result)

print("Predicted Class:", predicted_class)
