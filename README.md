# ClassificationAlgorithms
   
####Objective
The purpose of this project is to implement various classifiers for sparse high-dimensional vectors. The project consists of multiple components that involve using different representations for the dataset, implementing three different classifier models, assessing the performance of each classifier with the different representations, and analyzing some of the models that were estimated.    
   
####Getting the dataset
The dataset for this assignment is derived from the "Twenty Newsgroups Data Set" that is available at the UCI Machine Learning Repository. The dataset is already preprocessed and given to you in the form of its TF (term-frequency) representation for two types of features: bag-of-words and 5-character ngrams. You can download the dataset files from this link. There is also a readme file that describes what each file contains.   
   
####Different representations for the vectors
You need to run each classifier on both bag-of-words and 5-character ngrams representations, in which the weight of each term is determined by the following approaches:    
* Binary representation. The weight of each term that is present will simply be set to 1, irrespective of its actual term frequency.
* TF representation. The weight of each term will simply be its TF. These are the values that are provided in the files.
* Square-root of TF. The weight of its term will be the square-root of its TF.
* TF-IDF representation. The weight of each term will be the product of its TF and its inverse document frequency (IDF). See below on how to compute the IDF of a term.
* Binary-IDF representation. The weight of each term will be its IDF.
* Square root of TF with IDF. The weight of its term will be the product of its sqrt(TF) and its IDF.   
The IDF for a given term (IDF(t)) is given by   
```
IDF(t) = log2 (N / |D_t|)
```
   
where N is the total number of documents, and |D_t| is the number of documents that contain term t. NOTE: The IDF of a term is computed using the training set only and then you should use these IDF values on the validation/test set. For the features that occur in the validation/test set but do not occur in the training set, you can either set their IDF values to be a small constant, e.g., 1, or to 0.   

####Classification approaches
You need to develop three classifier approaches:  
* Centroid-based classifier.
* KNN Classifiers

Since there are 20 classes for this dataset, you will need to implement 20 binary classifier models for each classifier using a one-vs-rest approach.   
Your program should take as input the vector-space representation of the documents, both train and test sets, the class labels of them, the name of the classifier, and the output file name. Your program should first transform the vector space according to the feature-representation-option and then normalize each document vector to be of unit length. Your program should then train the classifier using the train set and then use the test set to evaluate the performance of your learned classifier. To assess the classifier performance on the test set, you should evaluate it in terms of the maximum F1(+ve) score. Upon completion, your program should write the classification solution on the test set to the output file, and report the quality of the results to the standard output, in terms of F1(+ve) score.    
Here is a sample command line for your program:   
```
classifier-name input-file input-rlabel-file train-file test-file class-file features-label-file feature-representation-option output-file [options]
```
