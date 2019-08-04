<h1>MNIST handwritten digits classification - Convolutional Neural Network</h1>

The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

<h2>Dependencies</h2>
<ul>
  <li>Tensorflow</li>
  <li>Matplotlib</li>
  <li>Numpy</li>
  <li>OpenCV</li>
</ul>

  Use <b>pip</b> to install any missing dependencies.

 <b>Keras:</b> a high-level API, to build and train models<br/>
 <b>tf.keras</b> is TensorFlow's implementation of the Keras API specification.
 
 <h2>Code-Walkthrough</h2>
 
 [mnist.py]
 1. Importing required libraries.
 2. Loading MNIST dataset
 3. Creating model and adding layers.
 4. Adding optimizer and loss function.
 5. Training the model.
 6. Saving the model.
 7. Loading the model.
 8. Generating output predictions.
 
 [input_img.py]
 1. Importing required libraries.
 2. Adding image(input image) path.
 3. Reshaping input image (28x28, similar to MNIST dataset).
 4. Loading pre-trained model.
 5. Generating output predictions.
