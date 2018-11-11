# Neural-Style-Transfer
Implemention of a neural style transfer algorithm where I optimized cost functions to get pixel values and an Image Classifier to determine if the image gotten from the neural style transfer is good artistically or not. 

Neural style transfer merges two images, namely, a content image and a style image, to create a generated image. The generated image combines the content of the image with the style of image. We used a previously trained convolutional network and built on top of it using transfer learning. I used a 19-layer version of the VGG network. The model was already trained on a very large ImageNet database and thus had learned to recognize a variety of low-level features (at the earlier layers) and high-level features (at the deeper layers).

I built the neural style transfer by implementing various functions used to determine the following:
	•	Content cost
	•	Style cost
	•	Gram matrix/Style matrix
	•	Total cost
A noisy image is then generated from the content image and its pixel values are now adjusted to match that of the content image and style image by using values gotten from our cost functions.

The second part is to predict if the image is good artistically or not, and this was done by in building an image classifier using Keras.
