# Ultrasound-Nerve-Segmentation

Introduction
---
In this project, we propose an algorithm for
automatic segmentation of nerves in ultrasound images of
the neck. Deep convolutional neural networks, when
trained end-to-end, pixels-to-pixel, have demonstrated
incomparable performance for various tasks such as image
classification and object detection. Image segmentation is a
low-level vision task which involves pixel level
classification. Recent techniques which used fully
convolutional neural networks for semantic segmentation
have limited the scope for Image Segmentation. In this
project, we demonstrate the use of the U-Net convolutional
architecture for ultrasound nerve segmentation.

Dataset
---
The dataset contains 5635 training images with their
correspomding masks and 5508 test images who masks are
supposed be predicted. The data is sourced from the Kaggle,
and similarly sized test dataset. The test dataset would be used
to detect the presence of Brachial Plexus nerve in the images
in an automated manner.

Preprocessing
---
The dataset consists of 5635 training images from 47 patients
and their corresponding masks. The testing data consists of
5508 images which are to be segmented. It is important to
separate the training images and their masks in different
arrays. The images were resized from 480*520 to a size of
64*80. It is important to keep all the patient data together with
a view to avoid overfitting. Image data from 42 patients was
used to trained the network. Images of 5 patients were kept as
validation. The test data is the unseen data and validated will
predict the mask for the test images. The preprocessing also
consisted of normalization of data to avoid imbalance in
weight updates. Image pixel values were between [0,1]
indicating grayscale and mask pixels were converted to 0 or 1
indicated segmented image. The 3-fold cross validation used
initially was not used later with a view to train on a lager
dataset and restrictions on computing power.

Conclusions
---
Neural networks are a powerful tool for data classification.
Convolutional Neural Networks can be used for Image
classification. The same technique can be extended to Image
Segmentation which is nothing but pixel-wise classification.
The U-Net architecture is a very useful technique for
biomedical image segmentation due to the complexity of
biomedical images. However, due to quality and amount of
training data and constraints on computing power, the
achieved value of dice coefficient is low. However, with better
computing power and optimization of the neural network the
Images can be better segmented and construction of a model
with greater accuracy will be possible. The only downside of
convolutional neural networks is that it requires huge training
data and large training time.
