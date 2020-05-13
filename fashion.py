
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
	print(tf.__version__)

	fashion_mnist = keras.datasets.fashion_mnist

	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	#the classification of the different items of clothing
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	#processing the data to make the images pixels
	plt.figure()
	plt.imshow(train_images[0])
	plt.colorbar()
	plt.grid(False)
	plt.show()


	#compressing the images
	train_images = train_images / 255.0

	test_images = test_images / 255.0

	#testing to make sufre the data is in the correct format

	plt.figure(figsize=(10,10))
	for i in range(25):
	    plt.subplot(5,5,i+1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(train_images[i], cmap=plt.cm.binary)
	    plt.xlabel(class_names[train_labels[i]])
	plt.show()

	#building the various layers of the network
	#the input nodes will be 784 input pixels from the 28 x 28 pixel image
	#the middle layer is to find correlations so the AI can find like pants sleeves and ect
	#the 3rd layer is the output layer where the label # corresponds to the label in clothing names

	model = keras.Sequential([
	    keras.layers.Flatten(input_shape=(28, 28)),
	    keras.layers.Dense(128, activation='relu'),
	    keras.layers.Dense(10)
	])

	#compile the model
	model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	#train the model from the training set of data
	model.fit(train_images, train_labels, epochs=8)

	#see how accurate the model is 
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print('\nTest accuracy:', test_acc)




