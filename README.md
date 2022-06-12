# Stuffy-ML
In this project for the part of machine learning, we use object recognition tensorflow to detect objects that are retrieved with objects that we have created previously, for this we use CNN as a construct and training model. We also use some other tools like TFLite for compressed size model and Keras.

We use a dataset from https://www.kaggle.com/datasets/zalando-research/fashionmnist and we have 10 class for the category of fashion. For the machine learning model, we use keras sequential architecture and add some additional layer. The convolutional front-end, we started with a convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer. And dor the second layer we use convolutional layer with a small filter size (3,3) and a modest number of filters (64) followed by a max pooling layer. The filter maps can then be flattened to provide features to the classifier.
All layers will use the ReLU activation function and the He weight initialization scheme, both best practices. Then we use [28, 28, 1] for  input_shape and get better testing result, that is 90% accuracy.

And then we convert the model to tensorflow lite format, so its can deploy on Android Apps.

We also test that tensorflow lite model, and get 90% accuracy

In the dataset, we distinguish between the following clothing objects:

1.  T-shirt/Top
2.  Trousers
3.  Pullover
4.  Dress
5.  Coat
6.  Sandal
7.  Shirt
8.  Sneaker
9.  Bag
10. Ankle Boot


