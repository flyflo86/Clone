import csv
import cv2
import numpy as np
#import keras
from os.path import split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt


batch_size_generator=128


def display_image(image):
#    cv2.imshow('image',image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    plt.figure()
    plt.imshow(image)

def prepro_image(image):
    # cropping the image (upper and lower part don't contain important information)
    adapted_image = image[50:140,:,:]
    # apply subtle blur
    adapted_image = cv2.GaussianBlur(adapted_image, (3,3), 0)
    # Scaling to shape input of model is expecting (200x66)
    adapted_image = cv2.resize(adapted_image,(200, 66), interpolation = cv2.INTER_AREA)
    # Converting to YUV 
    adapted_image = cv2.cvtColor(adapted_image, cv2.COLOR_RGB2YUV)
    return adapted_image


#*****************************DATA PREPARATION*******************************************
lines=[]
#with open('../data/driving_log.csv') as file:
with open('data_captured/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Current Path
def current_path(image_direction, line):
    source_path = line[image_direction]
    filename = split(source_path)[1]
    current_path = './data_captured/IMG/' + filename
#   current_path = '../data/IMG/' + filename
    return current_path     

def generate_training_path_sources(lines):
    image_paths, angles = [], []
    for line in lines:
        current_path_center = current_path(0,line)
        current_path_right = current_path(1,line)
        current_path_left = current_path(2,line)
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.15 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        image_paths.append(current_path_center)
        image_paths.append(current_path_left)
        image_paths.append(current_path_right)
        angles.append(steering_center)
        angles.append(steering_left)
        angles.append(steering_right)
        
    return image_paths, angles

#*********************Training Data Generator*************************************************
#Reading Data from Training Simulator and adding side camera images
def generate_data_training(image_paths, angles, batch_size_generator):
    image_paths, angles = shuffle(image_paths, angles)
    images,measurements = ([],[])
    while True:       
        for i in range(len(angles)):
            image = cv2.imread(image_paths[i])
            angle = angles[i]
            image = prepro_image(image)
            images.append(image)
            measurements.append(angle)
            if len(measurements) == batch_size_generator:
                yield (np.array(images), np.array(measurements))
                images, measurements = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
            # Data Augmentation: if steering angle is above a certain threshold, the image is flipped horizontally and also added to the training data
            if abs(angle) > 0.33:
                image = cv2.flip(image, 1)
                angle *= -1
                images.append(image)
                measurements.append(angle)
                if len(measurements) == batch_size_generator:
                    yield (np.array(images), np.array(measurements))
                    images, measurements = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)

def generate_data_training_visual(image_paths, angles, batch_size_generator):
    image_paths, angles = shuffle(image_paths, angles)
    images,measurements = ([],[])
    while True:       
        for i in range(len(angles)):
            image = cv2.imread(image_paths[i])
            angle = angles[i]
            image = prepro_image(image)
            images.append(image)
            measurements.append(angle)
            if len(measurements) == batch_size_generator:
                return (np.array(images), np.array(measurements))

            # Data Augmentation: if steering angle is above a certain threshold, the image is flipped horizontally and also added to the training data
            if abs(angle) > 0.10:
                image = cv2.flip(image, 1)
                angle *= -1
                images.append(image)
                measurements.append(angle)
                if len(measurements) == batch_size_generator:
                    return (np.array(images), np.array(measurements))

   
#**********Splitting Data in Train/Test*******************
image_paths, angles = generate_training_path_sources(lines)
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles, test_size=0.2, random_state=42)
images, measurements = generate_data_training_visual(image_paths,angles,batch_size_generator)

size_train=np.array(image_paths_train).shape[0]       
size_test=np.array(image_paths_test).shape[0] 
                                                                           
print('Train Dataset size:', np.array(image_paths_train).shape, np.array(angles_train).shape)
print('Test Dataset size:', np.array(image_paths_test).shape, np.array(angles_test).shape)
print('Image SHAPE:', images[0].shape)


#*********************************Visualizing DATA******************************************
#display_image(images[1])








#************************Model*********************************************************
Training_Mode=False    
if Training_Mode==True:
    

    
    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0, input_shape=(66,200,3)))
    #model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    #************************Training the Model**********************************************
    epochs=40
    #Compile the Model
    model.compile(loss='mse', optimizer='adam')
   # initializing Generators
    train_data_gen = generate_data_training(image_paths_train, angles_train, batch_size_generator)
    val_data_gen = generate_data_training(image_paths_train, angles_train, batch_size_generator)
    test_data_gen = generate_data_training(image_paths_test, angles_test, batch_size_generator)

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    history_object = model.fit_generator(train_data_gen, validation_data=val_data_gen, samples_per_epoch = size_train, nb_epoch = epochs, verbose=1, callbacks=[checkpoint], nb_val_samples=size_train)
    
    print('Test Loss:', model.evaluate_generator(test_data_gen, batch_size_generator))

    print(model.summary())
        
    model.save('model.h5')
    
    # Visualizing Predictions
    number_predictions = 6
    X_test, y_test = generate_data_training_visual(image_paths_test, angles_test, 6)
    y_pred = model.predict(X_test, number_predictions, verbose=2)
    
    
    
    #**********************Visualizing the training performance*******************************
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()    