
# coding: utf-8

# In[1]:


import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from sklearn.model_selection import train_test_split


# In[2]:


lines =[]
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' +filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


# In[3]:


def preprocess(img):
    new_images =[]
    numOfImage = len(img)
    for i in range(0, numOfImage):
        img1 = img[i]
        new_img = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
        new_images.append(new_img)
    
    new_images = np.array(new_images)
    return new_images


# In[ ]:


def addFlippedData(img, measurement):
    flipped_images=[]
    numOfImage = len(img)
    for i in range(0, numOfImage):
        img1 = img[i]
        image_flipped = np.fliplr(img1)
        flipped_images.append(image_flipped)
    
    flipped_images= np.array(flipped_images)
    
    img.append(flipped_images)
    
    measurement_flipped = -measurement
    measurement.append(measurement_flipped)
    
    return img, measurement


# In[4]:



X_train_proc = preprocess(X_train)


# In[ ]:





# In[5]:


import sklearn
def splitData(features, labels, splitRatio =0.2):

    X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size = splitRatio)
    trainSet =[X_train, y_train]
    validationSet = [X_validation, y_validation]

    return trainSet, validationSet
    


# In[6]:


model = Sequential()
model.add(Lambda(lambda x: (x/255.0)- 0.5, input_shape =(160, 320, 3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# In[16]:


def generator(samples, batch_size):
 # Create empty arrays to contain batch of features and labels#
    numFeatures = len(samples[0])
    batch_features = np.zeros((batch_size, 160, 320, 3))
    batch_labels = np.zeros((batch_size,1))
    while True:
        for offset in range(0, numFeatures, batch_size):
            endIndex = min(offset+ batch_size, numFeatures)
            batch_features = (samples[0][offset:endIndex])
            batch_labels = (samples[1][offset:endIndex])
        #for i in range(batch_size):
            # choose random index in features
         #   index= np.random.choice(len(features),1)
            #batch_features[i] = some_processing(features[index])
           # batch_features[i] = (features[index])
           # batch_labels[i] = labels[index]
            yield [batch_features, batch_labels]


# In[15]:


trainSet, validationSet = splitData(X_train_proc,y_train)


# In[ ]:


#model = Sequential()
#model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(1))
batch_size = 32
numTrain = len(trainSet[0])
numValidation = len(validationSet[0])
train_generator = generator(trainSet, batch_size)
validation_generator = generator(validationSet, batch_size)
model.compile(loss='mse', optimizer ='adam')
#model.fit(X_train_proc, y_train, validation_split =0.2, shuffle = True)
model_history = model.fit_generator(train_generator, samples_per_epoch=numTrain, nb_epoch=3)
model.save('model.h5')


# In[ ]:





# In[ ]:





# In[ ]:




