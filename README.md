# Semantic Segmentation

Self-Driving Car Engineer Nanodegree Program

## Introduction
In this project, we'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

|      um_000000.png       |        um_000017.png     |
:-------------------------:|:-------------------------:
![00](./writeup/um_000000.png)  |  ![17](./writeup/um_000017.png)
|      umm_000076.png      |        uu_000004.png     |
![76](./writeup/umm_000076.png) | ![04](./writeup/uu_000004.png)


## Write up

### Fully Convolutional Network

![FCN](./writeup/FCN.jpg)

* replace fully connected layers with one by one convolutional layers

* up-sampling through the use of transposed convolutional layers

* skip connections

### Convolution 1x1

```
conv_1x1 = tf.layers.conv2d(input_image,num_classes,1,1,padding='same',
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
```

### Transposed Convolutions

![TransposedConvolution](./writeup/TransposedConvolution.gif)

Blue maps are inputs,and cyan maps are outputs.No padding and no trides.

![upsampling](./writeup/up-sampling.png)

* First row (FCN-32s): The singlestream net, upsamples stride 32 predictions back to pixels in a single step. 

* Second row (FCN-16s): Combining predictions from both the final layer and the pool4 layer, at stride 16, lets our net predict finer details, while retaining high-level semantic information.  

* Third row (FCN-8s): Additional predictions from pool3, at stride 8, provide further precision.

### Skip Connections

![Skip](./writeup/Skip.png)

```
# 1x1 convolution of vgg layer7 output
# use random normal to initialize the kernel
# use L2 regularizer as the kernel regularizer

conv_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,1,padding='same',
                            kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# up-sampling,kernel size is (4,4),stride size is (2,2)
output = tf.layers.conv2d_transpose(conv_1x1,num_classes,4,2,padding = 'same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

# 1x1 convolution of layer4 output

layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out,num_classes,1,1,padding='same',
                                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
# skip connection
output = tf.add(output,layer4_conv_1x1)

```



### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Example Outputs
Here are examples of a sufficient vs. insufficient output from a trained network:

Sufficient Result          |  Insufficient Result
:-------------------------:|:-------------------------:
![Sufficient](./examples/sufficient_result.png)  |  ![Insufficient](./examples/insufficient_result.png)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Why Layer 3, 4 and 7?
In `main.py`, you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

### Optional sections
Within `main.py`, there are a few optional sections you can also choose to implement, but are not required for the project.

1. Train and perform inference on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). Note that the `project_tests.py` is not currently set up to also unit test for this alternate dataset, and `helper.py` will also need alterations, along with changing `num_classes` and `input_shape` in `main.py`. Cityscapes is a much more extensive dataset, with segmentation of 30 different classes (compared to road vs. not road on KITTI) on either 5,000 finely annotated images or 20,000 coarsely annotated images.
2. Add image augmentation. You can use some of the augmentation techniques you may have used on Traffic Sign Classification or Behavioral Cloning, or look into additional methods for more robust training!
3. Apply the trained model to a video. This project only involves performing inference on a set of test images, but you can also try to utilize it on a full video.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
