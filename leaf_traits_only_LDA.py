import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

np.random.seed(185)
split_random_state = 7
split = .9

root = '../input' 
train_data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train.csv'))
test_data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/test.csv'))

def encode(train, test):
    """
    Species is a string, so we have to encode it into numbers. Also prepare
    train and test data by removing 'species' and 'id' from train dataframe
    and also remove 'id' from test dataframe.
    """
    encoded = LabelEncoder().fit(train.species) 
    species = encoded.transform(train.species)           
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)    
    return train, species, test

train, species, test = encode(train_data, test_data)

sss = StratifiedShuffleSplit(species, 10, train_size=split, random_state=split_random_state)

for train_index, test_index in sss:
    traits_train, traits_test = train.values[train_index], train.values[test_index]
    species_train, species_test = species[train_index], species[test_index]

clf = LinearDiscriminantAnalysis()
clf.fit(traits_train, species_train)

predictions = clf.predict(traits_test)
acc = accuracy_score(species_test, predictions)

predictions = clf.predict_proba(traits_test)
ll = log_loss(species_test, predictions)

print ('accuracy = ', acc) #ended up at 0.98989
print ('log loss= ',ll)#ended up at 0.96249
