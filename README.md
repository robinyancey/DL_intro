I. What is Deep Learning?
-insert intro here please-
Associated Homework: Homework 1
1.1 Artificial Intelligence, Machine Learning, and Deep Learning
■ Artificial Intelligence
■ Machine learning
■ Learning representations from data
■ The “deep” in deep learning
■ Understanding how deep learning works, in three 
■ What deep learning has achieved so far
■ Don’t believe the short-term hype
■ The promise of AI 13
1.2 Before deep learning: a brief history of machine learning
■ Probabilistic modeling
■ Early neural networks
■ Kernel methods
■ Decision trees, random forests, and gradient boosting machines
■ Back to neural networks
■ What makes deep learning different
■ The modern machine-learning landscape
1.3 Why deep learning? Why now?
■ Hardware
■ Data
■ Algorithms
■ A new wave of investment
■ The democratization of deep learning
■ Will it last?


2. The mathematical building blocks of neural networks
-insert intro here please-
Associated Homework: Homework 1
2.1 A first look at a neural network
2.2 Data representations for neural networks
■ Scalars (0 D tensors)
■ Vectors (1D tensors)
■ Matrices (2D tensors)
■ 3D tensors and higher- dimensional tensors
■ Key attributes
■ Manipulating tensors in Numpy
■ The notion of data batches
■ Real-world examples of data tensors
■ Vector data
■ Time series data or sequence data
■ Image data
■ Video data
2.3 The gears of neural networks: tensor operations
■ Element-wise operations
■ Broadcasting
■ T ensor dot
■ T ensor reshaping
■ Geometric interpretation of tensor operations
■ A geometric interpretation of deep learning
2.4 The engine of neural networks: gradient-based optimization
■ What’s a derivative?
■ Derivative of a tensor operation: the gradient
■ Stochastic gradient descent
■ Chaining derivatives: the Backpropagation algorithm



3. Getting started with neural networks
-insert intro here please-
Associated Homework: Homework 1
3.1 Anatomy of a neural network
■ Layers: the building blocks of deep learning
■ Models: networks of layers
■ Loss functions and optimizers: keys to configuring the learning process
3.2 Introduction to Keras
■ Keras, T ensorFlow, Theano, and CNTK
■ Developing with Keras
3.3 Setting up a deep-learning workstation
■ Jupyter notebooks: the preferred way to run deep-learning experiments
■ Getting Keras running : two options
■ Running deep-learning jobs in the cloud: pros and cons
■ What is the best GPU for deep learning?
3.4 Classifying movie reviews: a binary classification example
■ The IMDB dataset
■ Preparing the data
■ Building your network
■ Validating your approach
■ Using a trained network to generate predictions on new data
3.5 Classifying newswires: a multiclass classification example
■ The Reuters dataset
■ Preparing the data
■ Building your network
■ Validating your approach
■ Generating predictions on new data
■ A different way to handle the labels and the loss
■ The importance of having sufficiently large intermediate layers
3.6 Predicting house prices: a regression example
■ The Boston Housing Price dataset
■ Preparing the data
■ Building your network
■ Validating your approach using K-fold validation



4. Fundamentals of machine learning
-insert intro here please-
Associated Homework: Homework 1
4.1 Three branches of machine learning
■ Supervised learning
■ Unsupervised learning
■ Self-supervised learning
4.2 Evaluating machine-learning models
■ Training, validation, and test sets
■ Things to keep in mind
4.3 Data preprocessing, feature engineering, and feature learning
■ Data preprocessing for neural networks
■ Feature engineering
4.4 Overfitting and underfitting
■ Reducing the network’s size
■ Adding weight regularization
■ Adding dropout
4.5 The universal workflow of machine learning
■ Defining the problem and assembling a dataset
■ Choosing a measure of success
■ Deciding on an evaluation protocol
■ repairing your data
■ Developing a model that does better than a baseline
■Scaling up: developing a model that overfits
■ Regularizing your model and tuning your hyperparameters
■ evaluation protocol
■ Preparing your data
■ Developing a model that does better than a baseline
■ Scaling up: developing a model that overfits
■ Regularizing your model and tuning your hyperparameters




5. Deep learning for computer vision
-insert intro here please-
Associated Homework: Homework 1
5.1 Introduction to convnets
■ The convolution operation
■ The max-pooling operation
5.2 Training a convnet from scratch on a small dataset
■ The relevance of deep learning for small-data problems
■ Downloading the data
■ Building your network
■ Data preprocessing
■ Using data augmentation
5.3 Using a pre-trained convnet
■ Feature extraction
■ Fine-tuning
5.4 Visualizing what convnets learn
■ Visualizing intermediate activations
■ Visualizing convent filters
■ Visualizing heatmaps of class activation


6. Deep learning for text and sequences
-insert intro here please-
Associated Homework: Homework 1
6.1 Working with text data
■ One-hot encoding of words and characters
■ Using word embeddings
■ Putting it all together: from raw text to word embeddings
6.2 Understanding recurrent neural networks
■ A recurrent layer in Keras
■ Understanding the LSTM and GRU layers
■ A concrete LSTM example in Keras
6.3 Advanced use of recurrent neural networks
■A temperature-forecasting problem
■ Preparing the data
■ A common-sense, non-machine-learning baseline
■ A basic machine-learning approach
■ A first recurrent baseline
■ Using recurrent dropout to fight overfitting
■ Stacking recurrent layers
■Using bidirectional RNNs to fight overfitting
■ Stacking recurrent layers
■ Using bidirectional RNNs
6.4 Sequence processing with convnets
■ Understanding 1D convolution for sequence data
■ 1D pooling for sequence data
■ Implementing a 1D convnet
■ Combining CNNs and RNNs to process long sequences

7. Advanced deep-learning best practices
-insert intro here please-
Associated Homework: Homework 1
7.1 Going beyond the Sequential model: the Keras functional API
■ Introduction to the functional API
■ Multi-input models
■ Multi-output models
■ Directed acyclic graphs of layers
■ Layer weight sharing
■ Models as layers
7.2 Inspecting and monitoring deep-learning models using
■ Keras callbacks and T ensorBoard
■ Using callbacks to act on a model during training
■ Introduction to T ensorBoard: the T ensorFlow visualization framework
7.3 Getting the most out of your models
■ Advanced architecture patterns
■ Hyperparameter optimization
■ Model ensembling
8. Generative deep learning
8.1 T ext generation with LSTM
■ A brief history of generative recurrent networks
■ How do you generate sequence data?
■ The importance of the sampling strategy
■ Implementing character-level LSTM text generation
8.2 DeepDream
■ Implementing DeepDream in Keras
8.3 Neural style transfer
■ The content loss
■ The style loss
■ Neural style transfer in Keras
8.4 Generating images with variational autoencoders
■ Sampling from latent spaces of images
■ Concept vectors for image editing
■ Variational autoencoders
8.5 Introduction to generative adversarial networks
■ A schematic GAN implementation
■ A bag of tricks
■The generator
■ The discriminator
■ The adversarial network
■ How to train your DCGAN




9. Conclusions
-insert intro here please-
Associated Homework: Homework 1
9.1 Key concepts in review
■ Various approaches to AI
■ What makes deep learning special within the field of machine learning
■ How to think about deep learning
■ Key enabling technologies
■ The universal machine-learning workflow
■ Key network architectures
■ The space of possibilities
9.2 The limitations of deep learning
■ The risk of anthropomorphizing machine-learning models
■ Local generalization vs. extreme generalization
9.3 The future of deep learning
■ Models as programs
■ Beyond backpropagation and differentiable layers
■ Automated machine learning
■ Lifelong learning and modular subroutine reuse
■ The long-term vision
9.4 Staying up to date in a fast-moving field
■ Practice on real-world problems using Kaggle
■ Read about the latest developments on arXiv
■ Explore the Keras ecosystem



*NOTE need to make sure these are covered somewhere
1. Introduction to Linux, Python and Jupyter hub
2. Using the Keras API for  ensorFlow
3. The T ensorflow session, and dynamic computational graph architectures 4. Natural language processing with
RNN and LSTM
5. Convolutional Neural Networks for classification
6. Convolutional Neural Networks for segmentation
7. Advanced computational graph models using core T ensorFlow
8. Generative models



