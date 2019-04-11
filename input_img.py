import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():

	imagePath = " # path for input image "
	image = cv2.imread(imagePath, 0)

	#reshaping input image (28x28, similar to MNIST dataset)
	IMG_SIZE = 28
	new_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
	new_img = tf.keras.utils.normalize(new_img, axis = 1)
	new_img1 = np.array([new_img])
	
	#loading pre-trained model
	model_ = tf.keras.models.load_model(" # path for pre-trained model ")
	predictions = model_.predict([new_img1])
	print(predictions[0])
	print(np.argmax(predictions[0]))

	plt.imshow(new_img1[0], cmap='gray')
	plt.show()
	

if __name__ == "__main__":
	main()