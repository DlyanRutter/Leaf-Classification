import os, keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.utils.visualize_util import plot
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, NumpyArrayIterator, array_to_img 
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras import backend as K

split_random_state = 7
split = .9
root = '../input'
np.random.seed(187)

def numeric_training(standardize=True):
    """
    Loads training data which is a 990x194 array AKA there are 194 columns and 990 rows
    where each row is a species instance and each column (other than the first two, which
    correspond to ID and species name) is a trait/feature. We then pop and return the ID and
    species name column to leave our dataset at 990x192 dimensions. We then encode the
    specieses and standardize the traits unless standardized is set to false.
    Returns ID, traits, and species.
    """
    #aquire data and put it into pandas.
    data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train.csv'))
    ID_number = data.pop('id')

    #first pop species column then use Label encoder's fit.transform from sklearn to encode labels
    species = data.pop('species')
    species = LabelEncoder().fit(species).transform(species)

    #data.values will return normal values (not standardized) in shape of an array
    traits = StandardScaler().fit(data).transform(data) if standardize else data.values

    #ID_number.shape = (990,), traits[0].shape=(192,), traits.shape=(990,192), species.shape=(990,)
    return ID_number, traits, species

def numeric_testing(standardize=True):
    """
    loads test data which already has species column removed. There are only 594 samples
    in the test set, so dimensions are 594x193. Pop ID number to make dimensions 594x192.
    Scale test data and return Id number tuple with shape (594,) and test data tuple with
    shape (594, 192)
    """
    test_data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/test.csv'))
    ID_number = test_data.pop('id')
    test_data = StandardScaler().fit(test_data).transform(test_data) if standardize else test_data.values
    return ID_number, test_data

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
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    Scales the image arrays and returns them
    """
    # Initialize the output array
    # Array has form (#_classes, length, width, channels) AKA (990, 96, 96, 1)
    X = np.empty((len(ids), max_dim, max_dim, 1))

    #iterate through images
    #enumerate returns tuple of (index, id)
    #each id comes with an image
    for i, idee in enumerate(ids):
        #load image into PIL format        
        #images are saved as 'path' + 'id_number" + ".jpg"
        #grayscale argument in load_img allows you to convert the image to gray scale
        x = resize_image(load_img(os.path.join(root,'/Users/dylanrutter/Downloads/images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)

        #tensorflow backbone so "channels last"
        #turn image into array with format (length, width, channels)
        x = img_to_array(x)
        
        # Corners of the bounding box for the image
        length = x.shape[0] #number of rows
        width = x.shape[1]  #number of columns

        if center:
            h1 = int((max_dim - length) / 2) 
            h2 = h1 + length 
            w1 = int((max_dim - width) / 2) 
            w2 = w1 + width 
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

def all_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted traits and image training data and
    does stratified shuffle split cross validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # ID, traits, species for training set
    #shape of traits is (990, 192), species is (990,) and ID is (990,)
    ID, traits, species = numeric_training()

    #images is an array of shape (990, 96, 96, 1)
    #corresponds to (num_training_images, dimension, dimension, channel)
    images = image_data(ID)
    
    # Cross-validation split and indexing
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(traits, species))

    #Designate testing and training sets
    traits_valid, image_valid, species_valid = traits[test_ind], images[test_ind], species[test_ind]  
    traits_train, image_train, species_train = traits[train_ind], images[train_ind], species[train_ind]

    #891 samples in train set from train data, 99 samples in valid set from train data
    #shape of traits_train is(891,192), image train is(891,96,96,1), species_train is (891,)
    #shape of traits_valid is (99, 192), image_valid is(99,96,96,1) species_valid is(99,)    
    return (traits_train, image_train, species_train), (traits_valid, image_valid, species_valid)

def all_test_data():
    """
    Loads traits image test data.
    Returns a tuple in the order ids, traits, image test data
    """
    ID, traits_test = numeric_testing()
    image_test = image_data(ID)
    #ID.shape = (594,), traits_test.shape = (594, 192), image_test.shape = (594, 96, 96, 1)
    return ID, traits_test, image_test

def categorize(class_vector, number_of_classes=None):
    """
    class_vector is a vector to be made into a matrix. number_of_classes is total number
    of classes. ravel input class vector so that you return a 1D array containing
    the same elements
    """
    class_vector = np.array(class_vector, dtype='int').ravel()
    if not number_of_classes:   #if no number_of_classes entry
        number_of_classes = np.max(class_vector) + 1
    n = class_vector.shape[0]
    binary_matrix = np.zeros((n, number_of_classes))
    binary_matrix[np.arange(n), class_vector] = 1
    return binary_matrix

#train and test tuples made by all_train_data
(traits_train, image_train, species_train), (traits_valid, image_valid, species_valid) =\
               all_train_data(random_state=split_random_state)


#train and test tuples made by all_train_data
(traits_train, image_train, species_train), (traits_valid, image_valid, species_valid) =\
               all_train_data(random_state=split_random_state)

#array of species train data with shape(891,)
#there are 891 samples in the train set
y = np.array(species_train).ravel() 

#maximum of raveled array
#in this case it is a value, not an array so no shape. Value = 99
#there are 99 species included, so the value is 99
num_classes = np.max(np.array(species_train).ravel()) + 1 

#in this case it is a value, not an array. Value = 891
#shape of y's 0th dimension
n = y.shape[0] 
 
#returns array filled with zeros of dimensions (n, num_classes). (891, 99)
#basically an array of zeroes ot the size representing (num_sumples, num_species)
categorical = np.zeros((n,num_classes))

#np.arange(n) will return a list of numbers between 0 and 891
categorical[np.arange(n), y] = 1

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
            #accessing taking array at each index/increment
            x = self.x[j]
            #randomly augments the single image 3D tensor
            x = self.image_data_generator.random_transform(x.astype('float32'))
            #apply the normalization configuration to the batch of inputs
            x = self.image_data_generator.standardize(x)
            #put x in proper location in initialized array
            batch_x[i] = x

        if self.save_to_dir:   #get every image in each batch
            for i in range(current_batch_size):
                
                #convert array back to image
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                
                #save image to directory                
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None: #no target data
            return batch_x
        batch_y = self.y[self.index_array] #get y values for batch
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


##Now to augment data using image generator 
imgen = image_data_generator_derivative(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

##training augmentor to take our image array and binary labels and generates batches of augmented data
imgen_train = imgen.flow(image_train, species_train_binary, seed=np.random.randint(1, 10000))
print('Data augmenter finished successfully!')

def multi_input_CNN_model():
    """
    Returns a Keras Functional API convoluted neural network model.
    This is a multi input model that will fit both our numeric and our
    image data. There are two convolution layers, two pooling layers, an image
    input layer, two activation layers, a numerical input layer, and two
    dense layers. The image array is flattened so that it can be
    concantenated with the numerical input layer. The first dense layer has a
    dropout of 0.4. The model is compiled with loss = categorical crossentropy,
    optimizer = rmsprop, and metrics = accuracy.
    """
    # Use same shape as our image tensor
    image = Input(shape=(96, 96, 1), name='image')

    #first activation function to use
    act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

    #first convolution layer, activation function is PReLU
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = act(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Pass it through the second convolutional layer
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    #Flatten our array so that we can combine it with our numerical data
    x = Flatten()(x)
    
    #Define the pre-extracted feature input
    #Input(shape=(192,)) means the expected input will be batches of 192-dimensional vectors
    numerical = Input(shape=(192,), name='numerical')
    
    # Concatenate the output of our convnet with our pre-extracted feature input
    concatenated = merge([x, numerical], mode='concat')

    #Add a fully connected layer
    #Dense is an integer that defines dimensionality of output space
    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.25)(x)

    #Get the final output
    #define dimensionality of output space as 99 and apply activation function 'softmax'
    out = Dense(99, activation='softmax')(x)

    # Functional API format defines model with 2 inputs
    # model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model = Model(input=[image, numerical], output=out)

    #Compiles model according to loss, optimizer, and metrics
    #tried adadelta optimizer, which was supposed to work well with images but it didn't do well
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

model = multi_input_CNN_model()
print('Model successfully created!')

def multi_input_generator(imgen, traits):
    """
#    A generator to train our keras neural network. It
#    takes the image augmenter generator and the array
#    of the pre-extracted features.
#    It yields a minibatch and will run until false
    """
    while True:
        for i in range(traits.shape[0]):
            # Get the image batch and labels
            batch_img, batch_y = next(imgen)
            
            # access the indices of the images that imgen gave us.
            x = traits[imgen.index_array]
            yield [batch_img, x], batch_y

#designate best model save location
#best_model_file = "fewer_epochs_leaf_history.h5" for 40 epochs
#best_model_file = 'leaf_history.h5' #for 150 epochs, loss = 0.00004, acc = 100%
#ModelCheckpoint will save model after every epoch into best_model_file
#only the best result based on 'val_loss' won't be overwritten
best_model_file = 'test_run_combo.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

#monitor='val_acc', mode='max' monitor accuracy
#monitor='val_loss', mode='min' monitor log loss

print('Training model...')

#model.fit_generator fits the model on data batch by batch by python generator.
#runs in parallel to the model for efficiency. 
history = model.fit_generator(multi_input_generator(imgen_train,traits_train),
                              samples_per_epoch=traits_train.shape[0],
                              nb_epoch=150,
                              validation_data=([image_valid, traits_valid], species_valid_binary),
                              nb_val_samples=traits_valid.shape[0],
                              verbose=0,
                              callbacks=[best_model])

print('Loading the best model...')

#load_model takes a filepath and returns a Keras model instance
model = load_model(best_model_file)
print('Best Model loaded!')

print(history.history.keys())
LABELS = sorted(pd.read_csv(os.path.join(root,'/Users/dylanrutter/Downloads/train.csv')).species.unique())
index, num_test, image_test = all_test_data()

species_pred = model.predict([image_test, num_test])
#print "This is from model.predict"
#print (species_pred)

species_pred_df = pd.DataFrame(species_pred,index=index,columns=LABELS)
#species_pred_df.tail()

def plot_figures(figures, nrows = 1, ncols=1, titles=False):
    """
    Plot a dictionary of figures. Figures = <title, figure> dictionary,
    nrows = number of rows of subplots for figure, ncols = number of columns
    of subplots for display.
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(sorted(figures.keys(), key=lambda s: int(s[3:]))):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        if titles:
            axeslist.ravel()[ind].set_title(title)
    for ind in range(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()
    if titles:
        plt.tight_layout()
    plt.show()

def get_dim(number):
    """
    Get the dimensions of a square-ish shape for plotting num images.
    """
    s = sqrt(number)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)

species_pred = np.argmax(species_pred, axis=1)
#print "This is from argmax"
#print (species_pred)

conv_layers = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)]
#print "This is conv_layers shape"
#print (conv_layers)

imgs_to_visualize = np.random.choice(np.arange(0, len(image_valid)), 1)
convout_func = K.function([model.layers[0].input, K.learning_phase()], [layer.output for layer in conv_layers])
conv_filters = convout_func([image_valid[imgs_to_visualize], 0])
#print "This is conv_imgs filts"
#print (conv_filters)

predictions = model.predict([image_valid[imgs_to_visualize],\
                             traits_valid[imgs_to_visualize]])
#print "This is validation predictions"
#print (predictions)

imshow = plt.imshow 

for im_count, im in enumerate(imgs_to_visualize):
    
    # Top 3 predictions
    top3_ind = predictions[im_count].argsort()[-3:]
    top3_species = np.array(LABELS)[top3_ind]
    top3_preds = predictions[im_count][top3_ind]
    
    # Actual leaf species
    actual = LABELS[species_valid[im]]
                    
    # Display the top 3 predictions and the actual species
    print("Top 3 Predicitons:")
    for i in range(2, -1, -1):
        print("\t%s: %s" % (top3_species[i], top3_preds[i]))
    print("\nActual: %s" % actual)

    # Show the original image
    plt.title("Image used: #%d (digit=%d)" % (im, species_valid[im]))
    imshow(image_valid[im][:, :, 0], cmap='gray')
    plt.tight_layout()
    plt.show()

    # Plot filter images
    for i, filters in enumerate(conv_filters):
        filters = filters[im_count]
        print("Convolutions Layer Image %d" % i)

        # Get it ready for the plot_figures function
        figure_dict = {'flt{0}'.format(i): filters[:, :, i] for i in range(filters.shape[-1])}
        plot_figures(figure_dict, *get_dim(len(figure_dict)))








"""
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]
"""

"""
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Log Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
"""

"""
#Function for making 3d array from image array

#x = np.asarray(x, dtype='float32')
#original np array has format (height, width)
#but original PIL image has format (width, height, channel)
#x = np.asarray(img)
#if len(x.shape) == 3:
#    if data_format == 'channels_first':
#        x.transpose(2,0,1)
#elif len(x.shape) == 2:
#    if data_format == 'channels_first':
#        x = x.reshape((1, x.shape[0], x.shape[1]))
#    else:
#        raise ValueError('unsupported image shape: ', x.shape)
#return x
"""
"""
#x = np.asarray(x, dtype='float32')
#original np array has format (height, width)
#but original PIL image has format (width, height, channel) so we reshape
#x = x.reshape((x.shape[0], x.shape[1], 1))
#x.shape
#length = x.shape[0]
#print length
"""











