# Recognizing food based on images (recipeshelf-ml-food-recognition)

Food image detection. This repo explores the state-of-art techniques available to detect food images and try to improve them.

### Current best result
Currently we are getting a *Top-1 accuracy of **88%*** and *top-5 accuracy of **97%*** on the dataset food-101 using **Inception-v3**. This is achieved by taking inspiration from this [Git Repo](https://github.com/stratospark/food-101-keras). But the author of this code ran the job on a very heavy system (Nvidia Titan X Pascal w/12 GB of memory, 96 GB of system RAM, as well as a 12-core Intel Core i7) that he built. Instead, we modified the code so that it runs on **Google Colab free tier** i.e. 12 GB system RAM, Nvidia K80 w/12 GB Memory and a chunk of shared computing.     

---
---

Instead of directly pulling out the big guns, we started with doing some EDA. The Exploratory Data Analysis notebook takes a random walk across several datasets and choosing which one to start with. Datasets explored were:

* UECFOOD100
* UECFOOD256
* food-101
* google-images

food-101 was chosen to start with over UECFOOD100 and UECFOOD256 as the later ones have a lot of duplicate data and many food items in a single image. This makes these datasets harder to start with, also almost all the food is Japanese. We thought of starting with a simpler dataset. Food-101 also, have already marked train and test set making easier to evaluate a standard performance.  

### Fundamental Machine Learning Techniques: *Modeling.ipynb*
Before trying obvious CNN I wanted to make sure I try fundamental ML techniques. I wanted to see the performance on just binary classification. As expected it wasn't very great. However, in doing so, we applied some good data analysis on the dataset. 

I converted the data to greyscale and resized to be square with a dimension 100 x 100 pixel.

![alt text](https://github.com/codingnuclei/recipeshelf-ml-food-recognition/blob/master/data/ReadmeStore/Carrotcake.jpg "Carrot Cake") ![alt text](https://github.com/codingnuclei/recipeshelf-ml-food-recognition/blob/master/data/ReadmeStore/donut.jpg "Donut")

Following were the accuracies for each of the algorithm on binary image classification.
* SVM - Accuracy: 61.2%, Precision: 60.5, Recall: 64.4%
* KNN - Accuracy: 59%, Precision: 61.6%, Recall: 47.6%
* Random Forest - Accuracy: 62.4%, Precision: 61.7% and Recall: 65.6%

Proving that we need to take out the big guns **Deep Learning**

### Inception-v3
First technique that is played with is **GoogLeNet** or **Inception V3** taking inpiration from this [Git Repo](https://github.com/stratospark/food-101-keras)