import tensorflow as tf
import numpy as np

# Note that these 3 fucntions are only used in the case of the CO-FCN model because it uses them as a pre-training phase, for the other model see the code of each one in the hybrid layers.

def increasing_trend_filter(f,input_length):
    
    '''
    Function to return a temporaty model that applies the custom increasing detection filter of length f.
    
    Args:
    
        f : length of filter (int), no default value
        input_length : length of input time series to create tensorflow model
    '''

    if f % 2 > 0:
        raise ValueError("Filter size should be even.")

    input_layer = tf.keras.layers.Input((input_length,1))

    output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=f,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
    output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

    model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

    filter_ = np.ones(shape=(f,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
    indices_ = np.arange(f)

    filter_[indices_ % 2 == 0] *= -1 # formula of increasing detection filter

    model.layers[1].set_weights([filter_]) # set the filter on the model

    return model

def decreasing_trend_filter(f,input_length):
    
    '''
    Function to return a temporaty model that applies the custom decreasing detection filter of length f.

    Args:
    
        f : length of filter (int), no default value
        input_length : length of input time series to create tensorflow model
    '''

    if f % 2 > 0:
        raise ValueError("Filter size should be even.")

    input_layer = tf.keras.layers.Input((input_length,1))

    output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=f,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
    output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

    model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

    filter_ = np.ones(shape=(f,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
    indices_ = np.arange(f)

    filter_[indices_ % 2 > 0] *= -1 # formula of increasing detection filter

    model.layers[1].set_weights([filter_]) # set the filter on the model

    return model

def peak_filter(f,input_length):

    '''
    Function to return a temporaty model that applies the custom peak detection filter of length f.

    Args:
    
        f : two times the sub filter length (int), no default value, the function takes as input 
            the length of two sub part of the outcom filter. The outcome filter
            has the length of f + f//2 (3 sub filters of length f//2)
        input_length : length of input time series to create tensorflow model
    '''

    if f % 2 > 0:
        raise ValueError("Filter size should be even.")

    input_layer = tf.keras.layers.Input((input_length,1))

    output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=f+f//2,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
    output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

    model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

    if f == 2:

        filter_ = np.asarray([-1,2,-1]).reshape((-1,1,1)) # if f == 2 then the filter will be [-1,2,-1]
    
    else:

        filter_ = np.zeros(shape=(f+f//2,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
        # filter_[0:f//2] = np.linspace(start=0,stop=1,num=f//2+1)[1:].reshape((-1,1,1))
        # filter_[f//2:f] = -1
        # filter_[f:] = np.linspace(start=filter_[f//2-1,0,0],stop=0,num=f//2+1)[:-1].reshape((-1,1,1))

        xmesh = np.linspace(start=0,stop=1,num=f//4+1)[1:].reshape((-1,1,1)) # define the xmesh used to construct the parabolic shape of the filter, the length of the xmesh is half the length of each sub part of the filter hence f//4

        filter_left = xmesh**2 # project the mesh on a parabolic function
        filter_right = filter_left[::-1] # inverse the filter_left to construct the right one

        # use filter_left and filter_right to constuct the outcome filter

        # the first part is in the negative part

        filter_[0:f//4] = -filter_left
        filter_[f//4:f//2] = -filter_right

        # the second part is in the positive part by with a double amplitude

        filter_[f//2:3*f//4] = 2 * filter_left
        filter_[3*f//4:f] = 2 * filter_right

        # the third part is in the negative part

        filter_[f:5*f//4] = -filter_left
        filter_[5*f//4:] = -filter_right

    model.layers[1].set_weights([filter_]) # set the filter on the model

    return model