# Leaf-Classification using convolutional neural networks with images only, convolutional neural networks with a combination of image data and numerical data, neural networks using only numerical data, and linear discriminant analysis on numerical data

# Some required modules for visualizing this code: TensorFlow, Keras, Sci-kit learn,
    # matplotlib, numpy, pandas, h5py

#all data can be downloaded at:    
#https://www.kaggle.com/c/leaf-classification/data

#leaf image_traits_combo_CNN.py: line 33, line 53, line 85 are each set up to
    #to load train/test/image files from my mac
    # e.g. line 33 is data = pd.read_csv(os.path.join(root, '/Users/dylanrutter/Downloads/train.csv'))
    # you'll have to type your own filepaths into each of those lines in place
    # of '/Users/dylanrutter/Downloads/train.csv'
    #PCs and Macs have different methods for file loading structure, so there
    #was no way for me to make a universal code at that part.
    #you'll need to do the same at lines 27 and 52 in "leaf_traits_only_NN.py"
    #you'll need to do the same at lines 28, 40, and 69 in "leaf_images_only_CNN.py"
    #you'll need to do the same at lines 15 and 16 in "leaf_traits_only_LDA.py"
