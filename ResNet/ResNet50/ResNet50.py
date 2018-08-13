# Import Library
from keras.models import Sequential, Model
from keras.engine.input_layer import Input
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Activation, Concatenate, ZeroPadding2D, add
from keras import backend as K

# Convolution and Batch Normalization Block
def residual_block(input_tensor, filters, strides=(2, 2)):

    """
    Utility function to apply 1 Convolutional Layer + Batch Normalization Layer + ReLu Activation Layer.
    Arguments:
        input_tensor: input tensor.
        filters: filter number in the Conv2D Layer.
        strides: strides in the Conv2D Layer.
    
    Returns:
        Output Activated tensor after applying skip connection to 3 Convolution Layers + Convolutioned Input Tensor.

    """
    
    f1, f2, f3 = filters
    ### Flag: Change to 1 when the input shape is channels_first
    bn_axis = 3
    
    # Conv Layer 1
    x = Conv2D(filters = f1, kernel_size = (1,1), strides = strides, padding = "valid")(input_tensor)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation("relu")(x)

    # Conv Layer 2
    x = Conv2D(filters = f2, kernel_size = (3,3), padding = "same")(x)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation("relu")(x)

    # Conv Layer 3
    x = Conv2D(filters = f3, kernel_size = (1,1), padding = "same")(x)
    x = BatchNormalization(axis = bn_axis)(x)

    shortcut = Conv2D(filters = f3, kernel_size = (1,1), strides = strides)(input_tensor)
    shortcut = BatchNormalization(axis = bn_axis)(shortcut)

    x = add([x, shortcut]) 
    x = Activation("relu")(x)

    return x

# Convolution and Batch Normalization Block
def residual_no_con_block(input_tensor, filters):

    """
    Utility function to apply 1 Convolutional Layer + Batch Normalization Layer + ReLu Activation Layer.
    Arguments:
        input_tensor: input tensor.
        filters: filter number in the Conv2D Layer.
    
    Returns:
        Output Activated tensor after applying  skip connection t0 3 Convolution Layers + Input Tensor.

    """
    
    f1, f2, f3 = filters

    ### Flag: Change to 1 when the input shape is channels_first
    bn_axis = 3
    # Conv Layer 1
    x = Conv2D(filters = f1, kernel_size = (1,1), strides=(1, 1), padding = "same")(input_tensor)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation("relu")(x)

    # Conv Layer 2
    x = Conv2D(filters = f2, kernel_size = (3,3), strides=(1, 1), padding = "same")(x)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation("relu")(x)

    # Conv Layer 3
    x = Conv2D(filters = f3, kernel_size = (1,1), strides=(1, 1), padding = "same")(x)
    x = BatchNormalization(axis = bn_axis)(x)

    x = add([x, input_tensor]) 
    x = Activation("relu")(x)

    return x

# ResNet 50 model Architecture
def ResNet50(input_x, input_y, input_z, include_top=True):
    
    """
    ResNet 50  Model Function.
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
    x = ZeroPadding2D((3, 3))(model_input)
    x = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), padding = "valid")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    # Residual Blcok Layer 1
    x = residual_block(x, [64, 64, 256], strides = (1,1))
    x = residual_no_con_block(x, [64, 64, 256])
    x = residual_no_con_block(x, [64, 64, 256])
    
    # Residual Blcok Layer 2
    x = residual_block(x, [128, 128, 512])
    x = residual_no_con_block(x, [128, 128, 512])
    x = residual_no_con_block(x, [128, 128, 512])
    x = residual_no_con_block(x, [128, 128, 512])

    # Residual Blcok Layer 3
    x = residual_block(x, [256, 256, 1024])
    x = residual_no_con_block(x, [256, 256, 1024])
    x = residual_no_con_block(x, [256, 256, 1024])
    x = residual_no_con_block(x, [256, 256, 1024])
    x = residual_no_con_block(x, [256, 256, 1024])
    x = residual_no_con_block(x, [256, 256, 1024])

        # Residual Blcok Layer 4
    x = residual_block(x, [512, 512, 2048])
    x = residual_no_con_block(x, [512, 512, 2048])
    x = residual_no_con_block(x, [512, 512, 2048])
    x = AveragePooling2D((7, 7), strides=(1, 1), padding='same')(x)
    
    if include_top:
        x = Dense(1000, activation='softmax')(x)
        resnet50 = Model(model_input, x)
        resnet50.load_weights("./pre_trained_model/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    else:
        x = Flatten()(x)
        resnet50 = Model(model_input, x)
        resnet50.load_weights("./pre_trained_model/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        
    print('ResNet_50 Pre-trained model succesfully loaded')    

    return resnet50

if __name__ == '__main__':
	# Set the input shape
	input_x, input_y, input_z = 224, 224, 3
	ResNet_50_model = ResNet50(input_x, input_y, input_z, include_top=False)

	
	"""
	Example to build model base on the pre_trained Inception V3 with weight.

	### include_top must be False in the Inception_V3() function.

	"""

	# Initiate model variable with the Inception v3's output
	model = ResNet_50_model.output

	# Create your architecture
	model = Dense(2048, activation='relu')(model)
	model = Dense(2048, activation='relu')(model)

	# Set the target_class_number according to the classes you want to train.
	target_class_number = 10
	prediction = Dense(target_class_number, activation='softmax', kernel_initializer='glorot_normal')(model)

	# Create model
	model = Model(inception.input, prediction)















