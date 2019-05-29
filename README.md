# Semantic Segmentation

### Discription
Labeling pixels on a road in images using a Fully Convolutional Network (FCN).

## Architecture / Rubic points

### Load pre-trained vgg model

Function ```load_vgg loads``` loads pre-trained vgg model.

### Learn the correct features from the images

The project has layers functions implemented. 

![alt text](./examples/3-Figure3-1.png "NN model")

### Optimize the neural network

The ```optimize``` function for the network is cross-entropy, and an Adam optimizer is used. 

```python
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

```

### Train the neural network

The ```train_nn``` function is implemented and prints time and loss per epoch/epochs of training.

### Train the model correctly with reasonable hyper parameters

The project trains model correctly, about 48s per epoch, 48sx40 epochs in total.

Final hyperparamters used for training.

```python
    L2_REG = 1e-5
    STDEV = 1e-2
    KEEP_PROB = 0.8
    LEARNING_RATE = 1e-4
    EPOCHS = 40
    BATCH_SIZE = 8
    IMAGE_SHAPE_KITI = (160,576)
    NUM_CLASSES = 2
```

### Correctly labeling the road

Results from the test images. From the GIF below, A pre-trained VGG-16 network combined with a fully convolutional network will successfully label the road. Performance was also improved  through the use of skip connections and adding element-wise to upsampled lower-level layers.

![Alt Text](./examples/video1.gif)


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

### QuickStart

##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

