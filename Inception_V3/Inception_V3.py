# Import Library
from keras.models import Sequential, Model
from keras.engine.input_layer import Input
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from keras import backend as K


# Convolution and Batch Normalization Block
def conv2d_bn_block(x, filters, kernel_size, padding='same', strides=(1, 1)):

    """
    Utility function to apply 1 Convolutional Layer + Batch Normalization Layer + ReLu Activation Layer.
    Arguments:
        x: input tensor.
        filters: filter number in the Conv2D Layer.
        kernel_size: convolution kernel size.
        padding: padding mode in the Conv2D Layer.
        strides: strides in the Conv2D Layer.
    
    Returns:
        Output Activated tensor after applying Convolutional Layer + Batch Normalization Layer .

    """
    
    ### Flag: Change to 1 when the input shape is channels_first
    bn_axis = 3
    
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias=False)(x)
    x = BatchNormalization(axis = bn_axis, scale=False)(x)
    x = Activation("relu")(x)
    return x
    

# First Inception Block in V3
def inception_block_1(x):
    
    """
    Inception Block Function 1 with 4 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 1 path1
    path1 = conv2d_bn_block(x, 64, (1,1))

    #Inception block 1 path2
    path2 = conv2d_bn_block(x, 48, (1,1))
    path2 = conv2d_bn_block(path2, 64, (5,5))

    #Inception block 1 path3
    path3 = conv2d_bn_block(x, 64, (1,1))
    path3 = conv2d_bn_block(path3, 96, (3,3))
    path3 = conv2d_bn_block(path3, 96, (3,3))

    #Inception block 1 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 32, (1,1))

    return Concatenate()([path1,path2,path3,path4])


# Second and Third Inception Block in V3
def inception_block_2(x):
    
    """
    Inception Block Function 2 with 4 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 2 path1
    path1 = conv2d_bn_block(x, 64, (1,1))

    #Inception block 2 path2
    path2 = conv2d_bn_block(x, 48, (1,1))
    path2 = conv2d_bn_block(path2, 64, (5,5))

    #Inception block 2 path3
    path3 = conv2d_bn_block(x, 64, (1,1))
    path3 = conv2d_bn_block(path3, 96, (3,3))
    path3 = conv2d_bn_block(path3, 96, (3,3))

    #Inception block 2 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 64, (1,1))

    return Concatenate()([path1,path2,path3,path4])


# Fourth Inception Block in V3
def inception_block_3(x):
    
    """
    Inception Block Function 3 with 3 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """    
    
    #Inception block 3 path1
    path1 = conv2d_bn_block(x, 384, (3,3), padding='valid', strides=(2, 2))

    #Inception block 3 path2
    path2 = conv2d_bn_block(x, 64, (1,1))
    path2 = conv2d_bn_block(path2, 96, (3,3))
    path2 = conv2d_bn_block(path2, 96, (3,3),  padding='valid', strides=(2, 2))

    #Inception block 3 path3
    path3 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    return Concatenate()([path1,path2,path3])

# Fifth Inception Block in V3
def inception_block_4(x):

    """
    Inception Block Function 4 with 4 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    #Inception block 4 path1
    path1 = conv2d_bn_block(x, 192, (1,1))

    #Inception block 4 path2
    path2 = conv2d_bn_block(x, 128, (1,1))
    path2 = conv2d_bn_block(path2, 128, (1,7))
    path2 = conv2d_bn_block(path2, 192, (7,1))

    #Inception block 4 path3
    path3 = conv2d_bn_block(x, 128, (1,1))
    path3 = conv2d_bn_block(path3, 128, (7,1))
    path3 = conv2d_bn_block(path3, 128, (1,7))
    path3 = conv2d_bn_block(path3, 128, (7,1))
    path3 = conv2d_bn_block(path3, 192, (1,7))

    #Inception block 4 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 192, (1,1))

    return Concatenate()([path1,path2,path3,path4])


# Sixth and Seventh Inception Block in V3
def inception_block_5(x):

    """
    Inception Block Function 5 with 4 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 5 path1
    path1 = conv2d_bn_block(x, 192, (1,1))

    #Inception block 5 path2
    path2 = conv2d_bn_block(x, 160, (1,1))
    path2 = conv2d_bn_block(path2, 160, (1,7))
    path2 = conv2d_bn_block(path2, 192, (7,1))

    #Inception block 5 path3
    path3 = conv2d_bn_block(x, 160, (1,1))
    path3 = conv2d_bn_block(path3, 160, (7,1))
    path3 = conv2d_bn_block(path3, 160, (1,7))
    path3 = conv2d_bn_block(path3, 160, (7,1))
    path3 = conv2d_bn_block(path3, 192, (1,7))

    #Inception block 5 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 192, (1,1))

    return Concatenate()([path1,path2,path3,path4])

# Eighth Inception Block in V3
def inception_block_6(x):

    """
    Inception Block Function 6 with 4 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 6 path1
    path1 = conv2d_bn_block(x, 192, (1,1))

    #Inception block 6 path2
    path2 = conv2d_bn_block(x, 192, (1,1))
    path2 = conv2d_bn_block(path2, 192, (1,7))
    path2 = conv2d_bn_block(path2, 192, (7,1))

    #Inception block 6 path3
    path3 = conv2d_bn_block(x, 192, (1,1))
    path3 = conv2d_bn_block(path3, 192, (7,1))
    path3 = conv2d_bn_block(path3, 192, (1,7))
    path3 = conv2d_bn_block(path3, 192, (7,1))
    path3 = conv2d_bn_block(path3, 192, (1,7))

    #Inception block 6 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 192, (1,1))

    return Concatenate()([path1,path2,path3,path4])


# Ninth Inception Block in V3
def inception_block_7(x):

    """
    Inception Block Function 7 with 3 paths.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 7 path1
    path1 = conv2d_bn_block(x, 192, (1,1))
    path1 = conv2d_bn_block(path1, 320, (3,3), padding='valid', strides=(2, 2))

    #Inception block 7 path2
    path2 = conv2d_bn_block(x, 192, (1,1))
    path2 = conv2d_bn_block(path2, 192, (1,7))
    path2 = conv2d_bn_block(path2, 192, (7,1))
    path2 = conv2d_bn_block(path2, 192, (3,3),  padding='valid', strides=(2, 2))

    #Inception block 7 path3
    path3 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    return Concatenate()([path1,path2,path3])


# Tenth and Eleventh Inception Block in V3

def inception_block_8(x):
    
    """
    Inception Block Function 8 with 4 paths and 4 branches.
    Arguments:
        x: input tensor.

    Returns:
        Output Concatenated tensor.
    """
    
    #Inception block 8 path1
    path1 = conv2d_bn_block(x, 320, (1,1))

    #Inception block 8 path2 and branches
    path2 = conv2d_bn_block(x, 384, (1,1))
    path2a = conv2d_bn_block(path2, 384, (1,3))
    path2b = conv2d_bn_block(path2, 384, (3,1))
    path2 = Concatenate()([path2a,path2b])

    #Inception block 8 path3 and branches
    path3 = conv2d_bn_block(x, 448, (1,1))
    path3 = conv2d_bn_block(path3, 384, (3,3))
    path3a = conv2d_bn_block(path3, 384, (1,3))
    path3b = conv2d_bn_block(path3, 384, (3,1))
    path3 = Concatenate()([path3a,path3b])

    #Inception block 8 path4
    path4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = conv2d_bn_block(path4, 192, (1,1))

    return Concatenate()([path1,path2,path3,path4])


# Inception V3 model Architecture
def Inception_V3(input_x, input_y, input_z, include_top=True):
    
    """
    Inception V3 Model Function.
    Arguments:
        input_x: image row.
        input_y: image column.
        input_z: colour channels.
        include_top: True  - Use Default input size and target classes 
                     Fales - Be able to use custom input size and target classes
        
    Returns:
        Output: weight loaded keras model.
    """ 
    
    model_input = Input(shape=(input_x, input_y, input_z))
    x = conv2d_bn_block(model_input, 32, (3,3), padding='valid', strides=(2, 2))
    x = conv2d_bn_block(x, 32, (3,3), padding='valid')
    x = conv2d_bn_block(x, 64, (3,3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn_block(x, 80, (1,1))
    x = conv2d_bn_block(x, 192, (3,3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = inception_block_1(x)
    x = inception_block_2(x)
    x = inception_block_2(x)
    x = inception_block_3(x)
    x = inception_block_4(x)
    x = inception_block_5(x)
    x = inception_block_5(x)
    x = inception_block_6(x)
    x = inception_block_7(x)
    x = inception_block_8(x)
    x = inception_block_8(x)
    x = AveragePooling2D((8, 8), strides=(1, 1), padding='same')(x)
    
    if include_top:
        x = Dense(1000, activation='softmax')(x)
        inception = Model(model_input, x)
        inception.load_weights("./pre_trained_model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    else:
        x = Flatten()(x)
        inception = Model(model_input, x)
        inception.load_weights("./pre_trained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
        
    print('Inception_V3 Pre-trained model succesfully loaded')    

    return inception

if __name__ == '__main__':
	# Set the input shape
	input_x, input_y, input_z = 299, 299, 3
	inception_V3_model = Inception_V3(input_x, input_y, input_z, include_top=False)

	
	"""
	Example to build model base on the pre_trained Inception V3 with weight.

	### include_top must be False in the Inception_V3() function.

	"""

	# Initiate model variable with the Inception v3's output
	model = inception_V3_model.output

	# Create your architecture
	model = Dense(2048, activation='relu')(model)
	model = Dense(2048, activation='relu')(model)

	# Set the target_class_number according to the classes you want to train.
	target_class_number = 10
	prediction = Dense(target_class_number, activation='softmax', kernel_initializer='glorot_normal')(model)

	# Create model
	model = Model(inception.input, prediction)















