#!/usr/bin/env python
# coding: utf-8

# In[100]:


import os
import sys
import scipy.io
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)


# In[125]:


content_image = imageio.imread("/Users/chukaezema/Documents/style_transfer/images/uOttawa-building1.jpg")
imshow(content_image)


# In[126]:


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieving the dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshaping a_C and a_G 
    a_C_unrolled = tf.transpose(
    tf.reshape(a_C,(1, n_H, n_W, n_C)),
    name='transpose'
)
    a_G_unrolled = tf.transpose(
    tf.reshape(a_G,(1, n_H, n_W, n_C)),
    name='transpose'
)
    
    # computing the cost with tensorflow 
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content


# In[127]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))


# In[128]:


style_image = imageio.imread("/Users/chukaezema/Documents/style_transfer/images/web_IMG_3091.jpg")
imshow(style_image)


# In[129]:


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A, tf.transpose(A))

    
    return GA


# In[130]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = " + str(GA.eval()))


# In[131]:


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieving dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshaping the images to have shape (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1/(4*np.square(n_C)*np.square(n_H*n_W))) * tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    
    
    return J_style_layer


# In[132]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    print("J_style_layer = " + str(J_style_layer.eval()))


# In[133]:


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# In[134]:


def compute_style_cost(model, STYLE_LAYERS):

    # initializing the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Selecting the output tensor of the currently selected layer
        out = model[layer_name]

        # Setting a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Setting a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        a_G = out
        
        # Computing style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Adding coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


# In[135]:


def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = (alpha*J_content)+(beta*J_style)
    
    return J


# In[136]:


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


# In[137]:


# Resetting the graph
tf.reset_default_graph()

# Starting interactive session
sess = tf.InteractiveSession()


# In[138]:


content_image = imageio.imread("/Users/chukaezema/Documents/style_transfer/images/uOttawa-building1.jpg")
content_image = reshape_and_normalize_image(content_image)


# In[139]:


style_image = imageio.imread("/Users/chukaezema/Documents/style_transfer/images/web_IMG_3091.jpg")
style_image = reshape_and_normalize_image(style_image)


# In[140]:


generated_image = generate_noise_image(content_image)
imshow(generated_image[0])


# In[141]:


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# In[142]:


# Assigning the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Selecting the output tensor of layer conv4_2
out = model['conv4_2']

# Setting a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Setting a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
a_G = out

# Computing the content cost
J_content = compute_content_cost(a_C, a_G)


# In[143]:


# Assigning the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Computing the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)


# In[144]:


J = total_cost(J_content, J_style, alpha = 10, beta = 40)


# In[145]:


# defining optimizer 
optimizer = tf.train.AdamOptimizer(2.0)

# defining train_step 
train_step = optimizer.minimize(J)


# In[146]:


def model_nn(sess, input_image, num_iterations = 200):
    
    # Initializing global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Running the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Running the session on the train_step to minimize the total cost
        _ = sess.run(train_step)
        ### END CODE HERE ###
        
        # Computing the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Printing every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # saving last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


# In[147]:


model_nn(sess, generated_image)


# In[ ]:




