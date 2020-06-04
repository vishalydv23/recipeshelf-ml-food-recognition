# Food image recognition (recipeshelf-ml-food-recognition)

Food image detection. This repo explores the state-of-art techniques available to detect food images and try to improve them.

# Current best result

## EfficientNet-B4
### Top-1 Accuracy with TTA 91.71%, Top-1 Accuracy without TTA 91.22%, Top-5 Accuracy without TTA 98.63%
The notebooks may show something went wrong in browser, download it and run it locally. 
Link to Notebook: [Food101 EfficientNet-B4](https://github.com/vishalydv23/recipeshelf-ml-food-recognition/blob/master/notebook/modeling/fastai/Food101EfficientNetb4.ipynb)


## ResNet-152
### Top-1 Accuracy with TTA 91.21%, Top-1 Accuracy without TTA 90.68%, Top-5 Accuracy without TTA 98.49%
Link to Notebook: [Food101 ResNet152](https://github.com/vishalydv23/recipeshelf-ml-food-recognition/blob/master/notebook/modeling/fastai/fodd101ResNet152.ipynb)

## EfficientNet-B7
**EfficientNet-B7 hasn't been much played around with. The basic code structure is made available.**
Link to Notebook: [Food101 EfficientNet-B7](https://github.com/vishalydv23/recipeshelf-ml-food-recognition/blob/master/notebook/modeling/fastai/Food101_Efficientnet_b7.ipynb)

Rest of the Readme is old and will be updated soon. 
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

Proving that we need to take out the big gun: **Deep Learning**

### Inception-v3
First technique that is played with is **GoogLeNet** ([Research Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)) or **Inception V3** taking inpiration from this [Git Repo](https://github.com/stratospark/food-101-keras). The author of this code ran the job on a very heavy system (Nvidia Titan X Pascal w/12 GB of memory, 96 GB of system RAM, as well as a 12-core Intel Core i7) that he built. Instead, we modified the code so that it runs on **Google Colab free tier**.

**Google colab** helps us to write and execute python in the browser. We don't have to configure anything as almost everything come pre-configured and most importantly we get free access to GPUs. These GPUs however are shared so, its not necessary that we always get the access but almost all the time apart from once or twice I got the GPUs on colab.

We get a linux box running for a session which typically last for 12 hours if there is a job running in background. In the free tier of the instance we usually get 12.7 GB of system RAM, around 37 GB of disk space and most importantly NVIDIA K80 graphic card. Which is pretty awesome to run the job with higher batch sizes. I tried running the job on my 1070M and 980Ti Chips which are very powerful chips but these chips suffer from a lower RAM size.

Finally, one more very important thing to remember is to dock the google drive to your colab project. Which is very simple to do. This will help you to save the learning milestones to google drive otherwise the data go away once session ends. Or you can manually download the hdf5 files, which definitely doesn't sound convenient.    

```python
%%time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input

from keras.models import load_model

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math


K.clear_session()

n_classes = 101

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.4)(x)
x = Flatten()(x)
predictions = Dense(n_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='drive/My Drive/Deeplearning/model/model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('drive/My Drive/Deeplearning/log/model4.log')

def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004
lr_scheduler = LearningRateScheduler(schedule)

# mixing the old code into GoogleNet
batch_size = 64

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    zoom_range=[.8, 1],
    channel_shift_range=30,
    fill_mode='reflect')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'food-101/train/',  # this is the target directory
    target_size=(299, 299),  # all images will be resized to 299x299
    batch_size=batch_size,
    seed=42,
    class_mode='categorical')  

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'food-101/test/',
    target_size=(299, 299),
    batch_size=batch_size,
    seed=42,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    validation_steps=25250 // batch_size, # dividing total number of images in test set by batch size
    steps_per_epoch=75750 // batch_size, # dividing total number of images in train set by batch size
    epochs=32,
    callbacks=[lr_scheduler, csv_logger, checkpointer])
```

As we get the Colab session lasting for only 10-12 hours or you might get disconnected from the internet. You can restart to train from the last checkpoint you saved in your google drive. Simplest way is to rerun the whole code again but load the weights of the model from the checkpoint before fitting the data i.e. fit_generator().

```python
model.load_weights('drive/My Drive/Deeplearning/model/model4.08-0.71.hdf5')
```
 
#### Model Evaluation
Once you see the convergence. You can load the model and do the evaluation as described in the original code. We want to evaluate the test set using multiple crops. This is expected to raise the accuracy by 5% compared to single crop evaluation. It is common to use the following crops: Upper Left, Upper Right, Lower Left, Lower Right, Center. We also take the same crops on the image flipped left to right, creating a total of 10 crops.

Evaluating the result on a random Padthai image from the internet


![alt text](https://github.com/codingnuclei/recipeshelf-ml-food-recognition/blob/master/data/ReadmeStore/padthai.jpg "padthai")


```python
import matplotlib.image as matimg
from PIL import Image
import PIL

model = load_model('model address/model4.08-0.67.hdf5')

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]

def predict_10_crop(img, top_n=5, plot=False, preprocess=True, debug=False):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)),
        
        flipped_X[:299,:299, :],
        flipped_X[:299, flipped_X.shape[1]-299:, :],
        flipped_X[flipped_X.shape[0]-299:, :299, :],
        flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
        center_crop(flipped_X, (299, 299))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])
    
    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
    return preds, top_n_preds

# Resizing and cropping the input image 
allowedAspectratio = 1.3

img = Image.open('test image address')

mywidth = 400
wpercent = (mywidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
img.save('resizedimage.jpg')

im = Image.open('resizedimage.jpg')
width, height = im.size   # Get dimensions
if(height > width):
    if((height/width) > allowedAspectratio):
        extraHeight = height - (width * 1.3)
        top = extraHeight/2
        bottom = height - extraHeight/2
        im = im.crop((0, top, width, bottom))
        im.save('resizedimage.jpg')
else:
    if((width/height) > allowedAspectratio):
        extraWidth = width - (height * 1.3)
        left = extraWidth/2
        right = width - extraWidth/2
        im = im.crop((left, 0, right, height))
        im.save('resizedimage.jpg')

img_arr = img.imread("resizedimage.jpg")
prediction, topNPrediction = predict_10_crop(img_arr, top_n=5, plot=True, preprocess=False, debug=True)
```

![alt text](https://github.com/codingnuclei/recipeshelf-ml-food-recognition/blob/master/data/ReadmeStore/padthaiprediction.PNG "padthai")

Let's see what is the most common prediction in all 10 crops.

```python
counts = np.bincount(prediction)
mostCommonPrediction = np.argmax(counts)
print(mostCommonPrediction)

labelDictonary = train_generator.class_indices
print(list(labelDictonary.keys())[list(labelDictonary.values()).index(mostCommonPrediction)])
```

70
pad_thai <---------- This is Awesomeeeeeee!!!!!!

#### Performance Analysis

We have modified how the data is prepared to for making prediction on the test set. Original code required a big RAM size, we have optimized it to run on much smaller RAM size. We ran it on a laptop with 16GB RAM and it worked fine. 

```python
import os
from PIL import Image
import matplotlib.image as matimg
import keras.backend as K

path = 'food-101/test/'

topNPred = 5
topPred2D = np.zeros((25250,12))
top5Pred3D = np.zeros((25250,10,topNPred))
imageCount = 0

for r, d, f in tqdm(os.walk(path)):
    for file in f:
        fileName = os.path.join(r, file)
        className = fileName.split('/')[-1]
        className = className.split('\\')[0]
        try:
            imageResize(fileName)
            img_arr = matimg.imread("data/updatedImage.jpg")
            prediction, topNPrediction = predict_10_crop(img_arr, top_n=topNPred, plot=False, preprocess=False, debug=False)
            
            K.clear_session()
            # populating the 2D array
            topPred2D[imageCount][0] = classList.index(className) # actual class index from classes.txt file
            topPred2D[imageCount][1:-1] = prediction # prediction for all 10 crops
            counts = np.bincount(prediction) 
            mostCommonPrediction = np.argmax(counts)
            topPred2D[imageCount][-1] = mostCommonPrediction # most common value among all 10 predictions
        
            top5Pred3D[imageCount] = topNPrediction
        except:
            print(fileName)
            
        
        imageCount = imageCount + 1
```

Also, we are storing the predicted results on the test set in HDF5 files for later use. 

```python
import h5py

# Address to store the HDF5 file 
hdf5Path = r'..\..\data\processed\predictions.hdf5'

h5f = h5py.File(hdf5Path, 'w')
h5f.create_dataset('topPred2D', data=topPred2D)
h5f.create_dataset('top5Pred3D', data=top5Pred3D)
h5f.close()
```