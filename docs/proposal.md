# Project Proposal

## 1. Motivation & Objective

We aim to look at common augmentation techniques in sensor data and understand the effect they have on classification accuracy per target class, that is aim to understand not only how augmentation affects overall classification accuracy but also per class accuracy. 

## 2. State of the Art & Its Limitations

Currently no work looks at the the effects of how augmentation affects class accuracy in sensor data, recent work (Lecun Paper) has looked at how augmentation effects class based accuracy for images which we look to expand on. 

## 3. Novelty & Rationale

We aim to explore various augmentation techniques commonly used on sensor data and understand the possible bias these induce into the system, the Lecun paper provides a good framework for exploring this problem. 

## 4. Potential Impact

We believe this work could provide a better understanding of the effects of augmentation, data augmentation techniques are used extensively in most DL systems due to a lack of enough data and exploring the true effects it has on the system could have a large impact on influencing further exploration into data augmentation but we also believe it has the broader effect of providing insights to understand what the network is actually learning. 

## 5. Challenges

The following are the main identified challenges:

* Access to varied datasets to test hypothesis. 
* Possible ambiguity in the results and interpreting the results in the right way.

## 6. Requirements for Success

* Access to good datasets with available metrics on particular models. 
* Skills required are python programming and familiarity with a Deep Learning library we use Keras with a Tensorflow backend. 
* We dont estimate the need for a lot of compute to carry out this experiment at the moment.

## 7. Metrics of Success

* We look at the class based testing accuracy for all the datasets we test on. 
* Using the base accuracy without augmentation as a baseline we seek to compare class level accuracies for model trained with augmented data and understand its effects. 

## 8. Execution Plan

* Consider one Baseline Model (ConvLSTM) and train it without augmentation on one dataset. We consider the HAR dataset. This will serve as our baseline. 
* Perform a literature survery of common augmentation techniques sensor data, understand what is commonly used and pick the best augmentation strategies. 
* Train the same model with just one augmentation applied at a time. 
* Compare the accuracy of each class with the baseline model.
* Apply more than one augmentation to the data and retrain the model, estimate accuracies for these. 
* Repeat process with different models and datasets. 


## 9. Related Work

### 9.a. Papers

List the key papers that you have identified relating to your project idea, and describe how they related to your project. Provide references (with full citation in the References section below).

### 9.b. Datasets

List datasets that you have identified and plan to use. Provide references (with full citation in the References section below).

### 9.c. Software

List softwate that you have identified and plan to use. Provide references (with full citation in the References section below).

## 10. References

List references correspondign to citations in your text above. For papers please include full citation and URL. For datasets and software include name and URL.
