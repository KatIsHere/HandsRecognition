# HandsRecognition
A program written in Python, which can identify right and left hands on a given set of photos. 
It works, using a trained neural net.

The program that uses data from the sensor to detect right and left hands
placed on the table. 

Input data and test data are included in their respective folders

Included packages description:
Package               	  Description
-----------------------   --------------------------------------------------
CNN_model.py              implements a convolutional neural 
			  network and trains it on given set 
			  of images
image_read.py             reads images from their folders and 
			  turns theim into sets of training/testing data
recognising_pictures.py   uses trained in CNN model to recognize hands on 
			  bigger image
find_hands 		  implements functions for finding and higlighting 
			  hands clases
selective_search          finds areas, that are deemed suspicious(they may 
			  contain an object) and finds most plausible of them

Usage: package recognising_pictures.py implements image checking from folder 
Test189x110, other packages have needed functions/models implementation

Neded packages for python:
Package               Version
--------------------- -----------
h5py                  2.7.1
Keras                 2.1.6
numpy                 1.14.3
opencv-contrib-python 3.4.1.15
opencv-python         3.4.1.15
tensorflow            1.8.0

BUGS: program wiil work ONLY on python 3.5, because tensorflow works only on python 3.5

@Dubska Kateryna, 2018
