# **Behavioral Cloning** 


**Behavioral Cloning Project**


[//]: # (Image References)

[image_nvidia_architecture]: ./examples/nvidia_architecture.png "Nvidia architecture"
[image_normal_center]: ./examples/normal_center.jpg "Center lane driving"
[image_recovery_from_left]: ./examples/recovery_from_left.jpg "Recovery from left"
[image_recovery_from_right]: ./examples/recovery_from_right.jpg "Recovery from right"
[image_before_flipped]: ./examples/before_flipped.png "Original image"
[image_after_flipped]: ./examples/after_flipped.png "Flipped image"
[image_turn_right]: ./examples/turn_right.jpg "Turn right"
[image_turn_left]: ./examples/turn_left.jpg "Turn left"
[image_reverse]: ./examples/normal_reverse.jpg "Reverse driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used is the [Nvidia's architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which contains five convolutional layers and three fully connected layers. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. I especially focused on collecting and augmenting various cases of data to prevent overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and turning the corner. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach & Model Architecture

The overall strategy for deriving a model architecture was to start with the verified model. The only difference between the Nvidia model and my model is that I used RGB inputs instead of YUV inputs. Details of the architecture is shown below.
![alt text][image_nvidia_architecture]

In order to gauge how well the model was working, I split my image into 80% of training data and 20% of validation data, and checked mean squared error on the validation set.

On my first model, I found that my car failed to turn the corner. 
I gathered data from various cases other than center driving cases because I thought that this failure was due to the imbalance of the training data. For details of data is described below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_normal_center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back in the lane. 

![alt text][image_recovery_from_left]
![alt text][image_recovery_from_right]

I then recorded right and left turn data. The reason I added this turning the corner data is that the first model I trained with the training data above is failed at the corner.

![alt text][image_turn_right]
![alt text][image_turn_left]


To augment the data set, I also flipped images and angles thinking that this would be the reverse direction data. For example, here is an image that has then been flipped:

![alt text][image_before_flipped]
![alt text][image_after_flipped]

I also recorded three laps on track one using center lane driving with reverse direction because I thought the flipped data would not be exactly same with actual reverse driving data. Of course, these data are also included the flipping process.

![alt text][image_reverse]


After the collection process, I had 46,620 number of data points. First, I preprocessed this data by converting BGR format to RGB format because I used cv2.imread() functions and it returns BGR format. Next, I cropped the 60 pixels from the top and 20 pixels from bottom of images to reduce the effect of the background. Lastly, I normalized the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 determined experimentally. I tried 10 epochs first, but after 5 epochs the loss value is under 6e-4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
