"""
Reusing several of my own functions from capstone_tests.py
"""
import os, keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, NumpyArrayIterator, array_to_img 
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split_random_state = 7
split = .9
root = '../input'
np.random.seed(186)

def species_and_id_training():
    """
    Loads training data, encodes species from string to numeric, 
    returns tuple of ID number, species array.
    """
    data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train.csv'))
    ID_number = data.pop('id')
    species = data.pop('species')
    species = LabelEncoder().fit(species).transform(species)
    #ID_number.shape = (990,), species.shape=(990,)
    return ID_number, species

def id_testing(standardize=True):
    """
    loads test data and returns id_numbers in same manner as species_and_id_training
    note that test_data already has species column removed
    """
    test_data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/test.csv'))
    ID_number = test_data.pop('id')
    #ID_number.shape = (594,)
    return ID_number

def resize_image(img, max_dim=96):
    """
    Rescale the image so that the longest axis has dimensions max_dim
    """
    bigger, smaller = float(max(img.size)), float(min(img.size))
    scale = max_dim / bigger
    return img.resize((int(bigger*scale), int(smaller*scale)))

def image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    Array will have form (#_classes, length, width, color channels)
    Channels will be last because backbone is TensorFlow
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    X = np.empty((len(ids), max_dim, max_dim, 1))

    #enumerate returns tuple of (index, id)
    #each id comes with an image
    for i, idee in enumerate(ids):
        #load image into PIL format
        x = resize_image(load_img(os.path.join(root,'/Users/dylanrutter/Downloads/images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        
        # Corners of the bounding box for the image
        length = x.shape[0] #number of rows
        width = x.shape[1]  #number of columns

        if center:
            h1 = int((max_dim - length) / 2) #(96/63) becomes 1
            h2 = h1 + length #length =64
            w1 = int((max_dim - width) / 2) #should be 0
            w2 = w1 + width #still 96
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)

        # Insert into image matrix
        #i is the value of the index at the current iteration
        #h1:h2 creates a unit space vector of numbers between h1 and h2
        #w1:w2 creates a unit space vector of numbers between w1 and w2
        #0:1 creates a unit space vector of numbers between 0 and 1
        #substitute x array at location indexed at X[i, h1:h2, w1:w2, 0:1]
        X[i, h1:h2, w1:w2, 0:1] = x
        
    #X[0].shape is (96,96,1), x[0][0].shape is (96,1), x[0][0][0].shape is 1
    #shape of X is still(990, 96, 96, 1)  
    #Scale the array values so they are between 0 and 1
    return np.around(X / 255.0)

def image_train_data(split=split, random_state=None):
    """
    loads pre-extracted image training data and does stratified shuffle split
    cross validation. Returns tuples of (image train, species train) and (image test, species test)
    """
    ID, species = species_and_id_training()
    images = image_data(ID)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(images, species))
    image_train, species_train = images[train_ind], species[train_ind]
    image_valid, species_valid = images[test_ind], species[test_ind]
    return (image_train, species_train), (image_valid, species_valid)

def image_test_data():
    """
    loads image test data and testing IDs and returns ids and images
    """
    ID = id_testing()
    image_test = image_data(ID)
    return ID, image_test

def categorize(class_vector, number_of_classes=None):
    """
    class_vector is a vector to be made into a matrix. number_of_classes is total number
    of classes. will return a binary matrix of class_vector. 
    """
    #ravel input class_vector so that you return a 1D array containing the same elements
    class_vector = np.array(class_vector, dtype='int').ravel()
    #if no number_of_classes_entry
    if not number_of_classes:
        number_of_classes = np.max(class_vector) + 1
    n = class_vector.shape[0]
    binary_matrix = np.zeros((n, number_of_classes))
    binary_matrix[np.arange(n), class_vector] = 1
    return binary_matrix

#train and validation tuples made by image_train_data
(image_train, species_train), (image_valid, species_valid) =\
              image_train_data(random_state=split_random_state)

#binary form of (species train samples in training data train set after sss, samples in training data validation set)
#has shape (891, 99)
species_train_binary = categorize(species_train)

#binary (validation samples in training data test set after sss, samples in training data validation set) 
#has shape (99,99)
species_valid_binary = categorize(species_valid)

print('Training data successfully loaded!!!')

class np_array_iterator_derivative(NumpyArrayIterator):
    """
    Iterator that yields data from a numpy array.
    Arguments:
        x: Numpy array of input data
        y: Numpy array of target data
        batch_size: Integer size of batch
        shuffle: Boolean, whether to shuffle the data between epochs
        seed: Random seed for data shuffling
        save_to_dir: save pictures being yielded in a viewable format
        save_prefix: String prefix to use for saving sample images if save_to_dir is set
        save_format: format to use for saving sample images (if save_to_dir is set).
    This will give access to a self.index_array that we can use to index through self.y and self.y
    """
    
    def next(self):
        """
        Returns the next batch. self.index_generator yields (index_array[current_index:
        current_index + current_batch_size], current index, current batch_size) where
        index_array is np.arange(n) where n is the total number of samples in the dataset
        to loop over
        """
        with self.lock:
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        #use generator

        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
        #initialize an array for batch

        for i, j in enumerate(self.index_array):
            #accessing array at each index/increment
            x = self.x[j]
            #randomly augments the single image 3D tensor
            x = self.image_data_generator.random_transform(x.astype('float32'))
            #apply the normalization configuration to the batch of inputs
            x = self.image_data_generator.standardize(x)
            #put x in proper location in initialized array
            batch_x[i] = x

        if self.save_to_dir:
            #get every image in each batch
            for i in range(current_batch_size):
                
                #convert array back to image
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                
                #save image to directory                
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            #no target data
            return batch_x
        batch_y = self.y[self.index_array]
        #get y values for batch
        return batch_x, batch_y
    

class image_data_generator_derivative(ImageDataGenerator):
    """
    Will allow us to access indices the generator is working with
    Generates minibatches of image data with real-time data augmentation.
    Arguments:
        rotation_range: degrees (0 to 180)
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked in the range
            [1-z, 1+z]. A sequence of two can be passed instead to select this range
        horizontal_flip: whether to randomly flip images horizontally
        vertical_flip: whether to randomly flip images vertically
        fill_mode: points outside the boundaries are filled according to the given mode
            ('constant', 'nearest', 'reflect', or 'wrap'). Default is 'nearest'.
    """

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        """
        x is a numpy array of input data, and y is a numpy aray of labels. Calls
        NumpyArrayIterator2 AKA the iterator that yields data from a numpy
        array. Yields batches of (X,y) where X is a numpy array of image data and y
        is a numpy array of its corresponding labels
        """
        return np_array_iterator_derivative(
            x, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


#Now to augment data 
imgen = image_data_generator_derivative(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

#training augmentor to take our image array and binary labels and generate batches of augmented data
imgen_train = imgen.flow(image_train, species_train_binary, seed=np.random.randint(1, 10000))

print('Data augmenter finished successfully!')

def image_CNN_model():
    """
    Convolutional neural network being trained using images only.
    """
    
    # Define the image input
    # Use same shape as our image tensor
    image = Input(shape=(96, 96, 1), name='image')
    
    act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

    #first convolution layer
    #conv2d order is #filters, #convolutions, number of convolutions
    #activation function is PReLU
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = act(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Pass it through the second convolutional layer
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
    
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.25)(x)
    
    out = Dense(99, activation='softmax')(x)
    model = Model(input=[image], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

model = image_CNN_model()
print ('Model successfully created!')

def image_generator(imgen):
    """
    A generator to train our keras nueral network. It takes the image augmenter
    generator and yields a minibatch. Runs until false.
    """
    while True:
        for i in range(891):
            batch_img, batch_y = next(imgen)
            yield [batch_img], batch_y

best_model_file = "test_run_images.h5"
#best_model_file = "leaf_image_CNN.h5" for val_loss #value was 0.9065 at 150
#best_model_file = "leaf_image_CNN_acc.h5" #for accuracy, value was 0.73737 at 150

best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1,\
                             save_best_only=True)

print('training model...')

history = model.fit_generator(image_generator(imgen_train),
                              samples_per_epoch=image_train.shape[0],
                              nb_epoch=150,
                              validation_data=([image_valid], species_valid_binary),
                              nb_val_samples=image_valid.shape[0],
                              verbose=0,
                              callbacks=[best_model])
                              
print ('Loading the best model...')
model = load_model(best_model_file)
print('Best Model Loaded')










    
