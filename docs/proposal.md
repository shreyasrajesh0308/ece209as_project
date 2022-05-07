# Project Proposal

## 1. Motivation & Objective

Regularization is a commonly used technique to ensure better generalization in neural network models. Recent works show that Data Augmentation (DA) for images using ResNet50 has a detrimental effect on closer analysis, the accuracy of select classes has dropped (for example is the ‘barn spider’ class accuracy falls from 68% to 46% by introducing random crop DA during training). 

 Human Activity Recognition (HAR) is vital for smart assistive technologies for usage in healthcare, skill assessment, smart homes, and industries. Fitness trackers are commercially available to detect step count, calorie burn, heart rate tracking, and fall detection. Wearable activity recognition relies on combinations of sensors, such as accelerometers, gyroscopes, or magnetic field sensors. Activities are commonly classified using feature extraction on sliding windows followed by classification, and template matching approaches. Frequently used data augmentation for HAR includes Noise Injection, Dynamic Time Warping, and Time Cropping.[2], [4].

Due to numerous applications of activity recognition, it is crucial to be aware of the downfalls of these data augmentation techniques. 

## 2. State of the Art & Its Limitations

To our knowledge, none of the contemporary works study the effect of data augmentation techniques on class accuracy for human activity recognition. Recent work [1] has looked at how augmentation affects class-based accuracy for images which we look to expand for human activity recognition. The current state of the art models for human activity recognition are [2]: DeepConvLSTM, DeepConvLSTM Attention, Multi-Head Convolutional Attention [2]

## 3. Novelty & Rationale

We aim to explore various augmentation techniques commonly used in supervised learning on sensor data commonly used in human activity recognition and hope to understand the possible bias they induce into the system. 

## 4. Potential Impact

We believe this work could provide a better understanding of the effects of augmentation on HAR. Data augmentation techniques are used extensively in most deep learning systems to improve generalization and reduce overfitting. Exploring the effects on class-wise accuracy will help to enhance understanding of the effects it has on the system and could have a large impact on influencing further exploration into data augmentation. The work also has the broader effect of providing insights into what is being learned by the deep learning network.

## 5. Challenges

The following are the main identified challenges:

* Number of Human activity recognition datasets is fewer, limiting our study.
* Datasets for human activity recognition have different features requiring varied model architectures
* Interpreting the results of data augmentation on a per-class basis could be ambiguous.

## 6. Requirements for Success

* Skills required are python programming and familiarity with a Deep Learning library. We shall use Keras with a Tensorflow backend. 
* Adequate compute for training deep learning models. 


## 7. Metrics of Success

* Comparison of dataset augmentation techniques on HAR
* Make inferences based on the results for the HAR classification problem
* Deduce the best data augmentation technique and study its performance against models without augmentation. Specifically, the accuracy per class, as well as the effects of various data augmentation techniques, is studied. 

## 8. Execution Plan

* Consider a baseline Model and train it without augmentation on the HAR dataset. This shall serve as our baseline. 
* Perform a literature survey of common augmentation techniques sensor data, understand what is commonly used and pick the best augmentation strategies. 
* Train the same model with just unique augmentation applied at a time. 
* Compare the accuracy of each class of the newly created model with the baseline model.
* Apply more than one augmentation to the data and retrain the model followed by estimating accuracies. 
* Repeat process with different models and datasets. 
* Maybe trying to find Optimum policies for Augmentation

## 9. Related Work
Deep-Learning based Human Activity Recognition has been seeing a lot of recent developments and is well studied in the literature [4], [3]. To tackle the diverse problems that are distinct to the Sensor/Human Activity Recognition Dataset such as the amount of Dataset Fidelity, High Frequency/Random Noise Corruption, variation in temporal scales, and sampling frequencies that mask the original data [2], researchers in continuous pursuit of strategies that would help the Deep learning models to still capture distinguish between activities in presence of these activities[5]. This problem is compounded by the lack of relevant datasets which align closely with the rich, diverse data representations that sensors yield in real-time. Augmentation strategies are inevitably utilized to address this issue thereby increasing the amount of the dataset size available at one’s disposal for attempting to train complex data-hungry Deep Learning models. Hence, there is a strong incentive to conduct further investigation along the lines of analyzing Augmentation and its impact, implications, and performance benefits that could be potentially reaped while exploring the fairness implications of models in serving the predictions for a candidate dataset that might be randomly sampled from any of the class entities.


### 9.a. Papers

* The Effects of Regularization and Data Augmentation are Class Dependent
* Sensor Data Augmentation by Resampling for Contrastive Learning in Human Activity Recognition
* Sensor-Based Datasets for Human Activity Recognition – A Systematic Review of Literature
* Time Series Data Augmentation for Deep Learning: A Survey
* Multi-Scale Convolutional Neural Networks for Time Series Classification

### 9.b. Datasets
Datasets that are tentatively planned to be used are :
* UCI HAR- https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
VanKasteren,
* CASAS Kyoto- http://casas.wsu.edu/datasets/
* CASAS Aruba- http://casas.wsu.edu/datasets/
* CASAS MultiResident- http://casas.wsu.edu/datasets/
* mHealth- http://archive.ics.uci.edu/ml/datasets/mhealth+dataset
* Opportunity- 
https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition#:~:text=Data%20Set%20Information%3A-,The%20OPPORTUNITY%20Dataset%20for%20Human%20Activity%20Recognition%20from%20Wearable%2C%20Object,%2C%20feature%20extraction%2C%20etc)



### 9.c. Software

TensorFlow- https://www.tensorflow.org/
Python- https://www.python.org/
Jupyter Notebook- https://jupyter.org/
Scikit-learn -scikit-learn: machine learning in Python — scikit-learn 1.0.2 documentation

## 10. References

[1] Balestriero, Randall & Bottou, Leon & LeCun, Yann. (2022). The Effects of Regularization and Data Augmentation are Class Dependent. 
[2] Wang, Jinqiang and Zhu, Tao and Gan, Jingyuan and Chen, Liming and Ning, Huansheng and Wan, Yaping. (2021). Sensor Data Augmentation with Resampling for Contrastive Learning in Human Activity Recognition.
[3]E. De-La-Hoz-Franco, P. Ariza-Colpas, J. M. Quero and M. Espinilla, "Sensor-Based Datasets for Human Activity Recognition – A Systematic Review of Literature," in IEEE Access, vol. 6, pp. 59192-59210, 2018, doi: 10.1109/ACCESS.2018.2873502.
[4] Wen, Qingsong et al. “Time Series Data Augmentation for Deep Learning: A Survey.” IJCAI (2021).
[5] Cui, Zhicheng and Chen, Wenlin and Chen, Yixin. (2016). Multi-Scale Convolutional Neural Networks for Time Series Classification. 



