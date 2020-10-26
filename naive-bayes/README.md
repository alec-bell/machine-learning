# Naive Bayes Classifier

`naive-bayes.py` implements the Naive Bayes classification algorithm. It uses the training data to build a probabilistic model spanning each of the features in the examples (22 features total). It then uses these probabilistic models to calculate the probability of a label given a new feature vector using the following formula.

![formula](https://render.githubusercontent.com/render/math?math=\prod_{i=1}^{22}p_i^{x_i}(1-p_i)^{1-x_i})

Whichever label yields the greatest probability given the new point is the label that is returned from the prediction.

## Some Context on the Data

The training and test sets were created from the medical data on cardiac Single Proton Emission Tomography (SPECT) images of patients and each patient is classified into two categories: normal or abnormal. The database of 267 SPECT images sets (patients) was processed to extract features that summarize the original SPECT images.

## Analysis

After running the `NaiveBayes.fit` method on the training set, an error rate of `22.5%` was yielded on the test set. This isn't great, but probably stems from the fact that the algorithm assumes each feature is independent. We also could yield a better model if there was more comprehensive data as well.