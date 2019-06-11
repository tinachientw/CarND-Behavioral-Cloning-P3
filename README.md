# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---


In this project, I design, train, and test a convolutional neural network (CNN) to clone the driving behavior from sample images recorded from [Udacity's driving simulator](https://github.com/udacity/self-driving-car-sim). 

The goals / steps of this project are the following: Use the simulator to collect data of good driving behavior Build, a convolution neural network (using Keras) that predicts steering angles from images Train and validate the model with a training and validation set Test that the model successfully drives around the track without leaving the road


### Submission Files

My project submission includes the following files:

* ```model.py``` – the script used to create and train the final model.
* ```model.ipynb``` - the notebook used for development
* ```drive.py``` – the script provided by Udacity that is used to drive the car. I did not modify this script in any way.
* ```model.h5``` – the saved model file. It can be used with Keras to load and compile the model.
* ```report.pdf``` – written report, that you are reading right now. It describes all the important steps done to complete the project.
* ```video.mp4``` – video of the car, driving autonomously on the basic track.

All these files can be found in my project [repository on GitHub]

### Installation & Resources
* Anaconda Python 3.7
* Udacity [Carnd-term1 starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with miniconda installation
* Udacity [Car Simulation](https://github.com/udacity/self-driving-car-sim) on MacOC
* Udacity [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

### Quickstart

* Two driving modes:

Training: For user to take control over the car
Autonomous: For car to drive by itself

* Collecting data: User drives on track 1 and collects data by recording the driving experience by toggle ON/OFF the recorder. Data is saved as frame images and a driving log which shows the location of the images, steering angle, throttle, speed, etc. Another option is trying on Udacity data sample.

Approach
---

To have any idea to start this project, the CNN that was eventually used was based on [NVIDIA's End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper with different input image size and with dropout added to improve robustness. From the paper, data collection is the first important part. Per project requirement, data collection can only performed on Track 1. I drove about 4 laps around Track 1 by keyboard control to collect data. The driving wasn't extrememly smooth as actual driving. So I decided to use Udacity sample data as starting point.

### Understanding Data
There are 3 cameras on the car which shows left, center and right images for each steering angle.
<img src="./examples/camera.png">

views_plot

<img src="./examples/data_log.png">

After recording and save data, the simulator saves all the frame images in IMG folder and produces a driving_log.csv file which containts all the information needed for data preparation such as path to images folder, steering angle at each frame, throttle, brake and speed values.

driving_log

In this project, we only need to predict steering angle. So we will ignore throttle, brake and speed information.

### Model Architecture
The model has:

0. Two preprocessing layers, which I will describe later when talking about data.
1. Three convolutional layers with ```(5,5)``` kernels (24, 26 and 48 kernels per layer, correspondingly) with ```(2,2)``` strides, followed by
2. Two convolutional layers with ```(3,3)``` kernels (64 kernels per layer) with ```(1,1)``` strides, followed by
3. Three fully connected layers (with 100, 50 and 10 neurons, correspondingly), followed by
4. Output layer with one output neuron that controls the steering wheel.

I decided to use ELU (Exponential Linear Unit) activation, because [there is evidence](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf) that it can be slightly better than RELU. 

<img src="./examples/model.png">


### Training and Validation
In order to train a model, I used two generators – one for training and one for validation. Validation data generator was used to assess out-of-sample performance. Training generator was performing random data augmentation to improve generalization capabilities of the model, but validation generator was only performing preprocessing without doing any of the augmentation. 

When training the model as described in NVIDIA paper, I noticed that training error quickly was becoming smaller than validation error, which is the sign of overfitting. To reduce that I introduced dropout layers after each convolutional and each dense layer with the exception of the output layer. After training the model with different values of dropout I stopped at 0.5 for the final model.




### Data Preprocessing

The secret sauce in making the car drive itself well is not so much the architecture, but the data.


Here’s the original distribution of the training data (the x-axis corresponds to steering angles and the y-axis is the data point count for that angle range; Blue bars represent overrepresented classes, removed during data preprocessing step.

<img src="./examples/hist_aug.png">
<img src="./examples/hist_aug_2.png">


<img src="./examples/hist_depth10.png">
<img src="./examples/hist_depth3.png">


### Challenges with the Project

Originally I thought that ```steps_per_epoch``` and ```validation_steps``` parameters of ```.fit_generator()``` method require the number of training examples. When I provided numbers of training examples, the training went extremely slow even though I was using a high-end GPU for training. At first, I thought I was hitting the hard drive read bottleneck, because my hard drive is old and slow. I tried to solve this problem by pre-loading all cleaned data into memory and then using that loaded data to pass to train generator and perform augmentation on the fly. I think that sped things up but just a little bit. After some time of frustration I finally realized that I was using ```steps_per_epoch``` and ```validation_steps``` parameters all wrong. I then adjusted the values of these parameters and the training started to be as fast as I expected given the speed of my GPU. I learned my lesson and will never forget what these two parameters mean.

I used generators of Keras before, for example ```flow_from_directory()```, but I have never written my own custom generator. For some reason, I thought that it is too difficult and my level of Python was not advanced enough to write my own generators. I was mistaken and realized that generators are not that difficult. This was a really good practice and not only for Keras. Generators are widely used in Python and now I feel more confident in my ability to use and create them.

