import os, keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K

split_random_state = 7
split = .9
root = '../input'
np.random.seed(187)

def numeric_training(standardize=True):
    """
    Loads training data, formats to feature array, encodes label, scales features
    returns tuple of ID number, feature array, label array.
    If standardize is set to False, data won't be standardized.
    """
    #data in format 990 X 194 array. There are 194 columns and 990 rows where each row is a
    #species except for the first two, which correspond to id and species name
    #aquire data and put it into pandas.
    data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train.csv'))

    # Calling ID will return ID column, df now has 193 columns
    ID_number = data.pop('id')

    #species is text, so we need to encode
    #first pop species column then use Label encoder's fit.transform from sklearn to encode labels
    species = data.pop('species')
    species = LabelEncoder().fit(species).transform(species)

    #after popping species, data is of shape 990 X 192 and everything in data is numerical values
    #we can standarize by setting meant to 0 and std to 1
    #data.values will return normal values (not standardized) in shape of an array
    traits = StandardScaler().fit(data).transform(data) if standardize else data.values

    #ID_number.shape = (990,), traits[0].shape=(192,), traits.shape=(990,192), species.shape=(990,)
    return ID_number, traits, species

def numeric_testing(standardize=True):
    """
    loads test data and scales it then returns ID and test data
    """
    #test data has species column already removed,
    #only 594 samples in test set, so dimensions are 594x193
    #similar workings as load_numeric_training() function. See above.
    test_data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/test.csv'))
    ID_number = test_data.pop('id')
    test_data = StandardScaler().fit(test_data).transform(test_data) if standardize else test_data.values

    #ID_number.shape = (594,)
    #test_data.shape = (594, 192)
    return ID_number, test_data

def all_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted traits and image training data and
    does stratified shuffle split cross validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # ID, traits, species for training set
    #shape of traits is (990, 192), shape of species is (990,), shape of ID is (990,)
    ID, traits, species = numeric_training()

    # Cross-validation split and indexing
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(traits, species))

    #Designate training and validation
    traits_valid, species_valid = traits[test_ind], species[test_ind]  
    traits_train, species_train = traits[train_ind], species[train_ind]

    #891 samples in train set from train data, 99 samples in valid set from train data
    #shape of traits_train is(891,192), species_train is (891,)
    #shape of traits_valid is (99, 192), species_valid is(99,)    
    return (traits_train, species_train), (traits_valid, species_valid)

def all_test_data():
    """
    Loads traits image test data.
    Returns a tuple in the order ids, traits, image test data
    """
    ID, traits_test = numeric_testing()
    #ID.shape = (594,), traits_test.shape = (594, 192), image_test.shape = (594, 96, 96, 1)
    return ID, traits_test

def categorize(class_vector, number_of_classes=None):
    """
    class_vector is a vector to be made into a matrix. number_of_classes is total number
    of classes. will return a binary matrix of class_vector
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

#train and test tuples made by all_train_data
(traits_train, species_train), (traits_valid, species_valid) =\
               all_train_data(random_state=split_random_state)

#binary form of (species train samples in training data train set after sss, samples in training data validation set)
#has shape (891, 99)
species_train_binary = categorize(species_train)

#binary (validation samples in training data test set after sss, samples in training data validation set) 
#has shape (99,99)
species_valid_binary = categorize(species_valid)

def traits_NN_model():
    model = Sequential()
    model.add(Dense(100,init='uniform',input_dim=192))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5)) 
    model.add(Dense(100, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(99, init='uniform'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
 #   model.summary()


print('Training_model...')
best_model_file = "test_run_numerical.h5"
#best_model_file = "leaf_traits.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss',verbose=1,\
                             save_best_only=True)
model = traits_NN_model()
model.fit(traits_train, species_train_binary,
          batch_size=16,
          validation_data=(traits_valid,species_valid_binary),
          nb_epoch=150, callbacks=[best_model], verbose=0)
print('Loading the best model...')

model = load_model(best_model_file)
print('Best model loaded!')
####Best accuracy was 1.0000, and best val_loss was 0.0312











