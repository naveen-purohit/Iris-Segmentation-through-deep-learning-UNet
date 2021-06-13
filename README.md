# Iris-Segmentation-through-deep-learning-UNet
# U-Net Architecture
U-Net is a type of Convolution neural network  used for image segmentation. it contains encoder part for extracting low level features,decoder part for transforming low level features into high resolution images and skip connections to concatenate the low level features which high level.its architecture resembles to the letter U that is why it is called as U-net.here leftmost part is Encoder, right most part is Decoder and middle part is skip connections
![image](https://github.com/naveen-purohit/Iris-Segmentation-through-deep-learning-UNet/blob/main/unet%20architecture.png)
# Iris Segmentation
Automated iris segmentation is an important component of biometric identification. The role of artificial intelligence, particularly machine learning and deep learning, has been considerable in such automated delineation strategies. Although the use of deep learning is a promising approach in recent times, some of its challenges include its high computational requirement as well as availability of large annotated training data.We introduce an interactive variant of UNet for iris segmentation,to lower training time while improving storage efficiency through reduction in the number of parameters involved. The interactive component helps in generating the ground truth for datasets having insufficient annotated samples.

![image](https://github.com/naveen-purohit/Iris-Segmentation-through-deep-learning-UNet/blob/main/Screenshot%20(5).png)
# Dataset
 we have used here the IITD  iris dataset for the training purpose of U-net and as the label of training images we have used ground truth mask images contained in ground truth folder.
# Modules/Libraries installation requirements
#### Tensorflow (works as backend for keras)
#### keras (api for implementing U-net)
#### openCV
#### numpy
#### glob
# Evaluation
Neural Network model uses Accuracy metrics for computing the accuracy of model,Adam optimizer for optimizing the Loss/cost function upto its minimum value and Binary_crossentropy as a loss function because we want to predict the binary segmented mask of iris images.as we approach towards total no of epochs the accuracy of model increases while the loss decreases
![images](https://github.com/naveen-purohit/Iris-Segmentation-through-deep-learning-UNet/blob/main/Screenshot%20(7).png)
# Results
Dataset |Accuracy| Loss|test accuracy|test loss
 ---- | ----- | ------ | ------  | ------  
 IITD | 0.9876| 0.0295 |0.9834|0.0430 
# hyperparameters
we have given total of 896 images for training by dividing it in the batch size of 2,epochs given are 20 so the model iterate over the whole dataset 20 times until loss minimizes.
# Execution/Run
To load the training images and ground truth labels you can run
```
python util.py
```
To train the model, you can run
```
python UNet_iris_segmentation.py 
```
To test the model, you can run
```
python predict.py
```
