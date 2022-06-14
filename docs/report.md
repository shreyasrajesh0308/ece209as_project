# Report


## Table of Contents
* [Abstract](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#abstract)
* [Introduction](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#1-introduction)
* [Related Work](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#2-related-work)
* [Technical Approach](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#3-technical-approach)
* [Evaluation and Results](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#4-evaluation-and-results)
* [Discussion and Conclusions](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#5-discussion-and-conclusion)
* [References](https://github.com/shreyasrajesh0308/ece209as_project/blob/main/docs/report.md#6-references)



## Abstract
Presently, Human Activity Recognition (HAR) is an important task of  wearable devices and smartphones. These devices use a machine learning model to classify the user’s activity from the sensor data such as accelerometer and gyroscope. Due to a limitation of labeled HAR datasets, data augmentation is frequently utilized to enhance the model’s performance. In this work we explore the impact of data augmentation and weight decay on Human Activity Recognition datasets, specifically UCI-HAR, Pamap2, USC-HAD across LSTM, DeepConvLSTM model architectures. A similar work on image classification using ResNet architectures indicated the trend of  increase in overall model accuracy was met with a stark drop of accuracy in a select few classes of ImageNet. For HAR datasets, we do not find a significant drop in accuracy which might be attributed to fewer classes in HAR generally under twenty when compared to ImageNet with around thousand classes.


## 1. Introduction

Regularization is a commonly used technique to ensure better generalization in neural network models. Recent works show that Data Augmentation (DA) for images using ResNet50 has a detrimental effect on closer analysis, the accuracy of select classes has dropped (for example is the ‘barn spider’ class accuracy falls from 68% to 46% by introducing random crop DA during training). The work shows that there can be a massive risk of bias being introduced with many classes having a stark reduction in accuracy after augmentation in large image recognition models. The study noted that even in data agnostic techniques like weight decay, there is imbalance in the change of class accuracies.
 Human Activity Recognition (HAR) is vital for smart assistive technologies for usage in healthcare, skill assessment, smart homes, and industries. Fitness trackers are commercially available to detect step count, calorie burn, heart rate tracking, and fall detection. Wearable activity recognition relies on combinations of sensors, such as accelerometers, gyroscopes, or magnetic field sensors. Activities are commonly classified using feature extraction on sliding windows followed by classification, and template matching approaches. Frequently used data augmentation for HAR includes Noise addition, Scaling, Resampling, Magnify,Rotation, Time Warping, [2], [4]. Due to numerous applications of activity recognition, it is crucial to be aware of the downfalls of these data augmentation techniques.

State of the Art & Its Limitations \
To our knowledge, none of the contemporary works study the effect of data augmentation techniques on class accuracy for human activity recognition. Recent work [1] has looked at how augmentation affects class-based accuracy for images which we look to expand for human activity recognition. The current state of the art models for human activity recognition are [2]: DeepConvLSTM, DeepConvLSTM Attention, Multi-Head Convolutional Attention [2]

 Novelty & Rationale \
We aim to explore various augmentation techniques commonly used in supervised learning on sensor data commonly used in human activity recognition and hope to understand the possible bias they induce into the system.

 Potential Impact \
We believe this work could provide a better understanding of the effects of augmentation on HAR. Data augmentation techniques are used extensively in most deep learning systems to improve generalization and reduce overfitting. Exploring the effects on class-wise accuracy will help to enhance understanding of the effects it has on the system and could have a large impact on influencing further exploration into data augmentation. The work also has the broader effect of providing insights into what is being learned by the deep learning network.

 Challenges \
The following are the main identified challenges:
* Number of Human activity recognition datasets is fewer, limiting our study
* Interpreting the results of data augmentation on a per-class basis could be ambiguous

 Requirements for Success
* Skills required are python programming and familiarity with a Deep Learning library. We shall Keras with a Tensorflow backend
* Adequate compute for training deep learning models, models are trainable using google colab GPUs  

 Metrics of Success
* Comparison of dataset augmentation techniques on HAR
* Make inferences based on the results for the HAR classification problem
* Deduce the best data augmentation technique and study its performance against models without augmentation. Specifically, the accuracy per class, as well as the effects of various data augmentation techniques, is studied

 Execution Plan
* Consider a baseline Model and train it without augmentation on the HAR dataset. This shall serve as our baseline
* Perform a literature survey of common augmentation techniques sensor data, understand what is commonly used and pick the best augmentation strategies
* Train the same model with just unique augmentation applied at a time
* Compare the accuracy of each class of the newly created model with the baseline model
* Apply more than one augmentation to the data and retrain the model followed by estimating accuracies
* Repeat process with different models and datasets


## 2. Related Work
Deep-Learning based Human Activity Recognition has been seeing a lot of recent developments and is well studied in the literature [4], [3]. To tackle the diverse problems that are distinct to the Sensor/Human Activity Recognition Dataset such as the amount of Dataset Fidelity, High Frequency/Random Noise Corruption, variation in temporal scales, and sampling frequencies that mask the original data [2], researchers in continuous pursuit of strategies that would help the Deep learning models to still capture distinguish between activities in presence of these activities[5]. This problem is compounded by the lack of relevant datasets which align closely with the rich, diverse data representations that sensors yield in real-time. Augmentation strategies are inevitably utilized to address this issue thereby increasing the amount of the dataset size available at one’s disposal for attempting to train complex data-hungry Deep Learning models. Hence, there is a strong incentive to conduct further investigation along the lines of analyzing Augmentation and its impact, implications, and performance benefits that could be potentially reaped while exploring the fairness implications of models in serving the predictions for a candidate dataset that might be randomly sampled from any of the class entities.

## 3. Technical Approach

### a. Datasets
We consider the following 3 popular open source datasets
#### UCIHAR [12]
Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors
#### USC-HAD [11]
The focus of the dataset is healthcare related applications such as physical fitness monitoring The activity data is captured by a high-performance inertial sensing device and includes 12 activities and collected data from 14 subjects
#### PAMAP2 [10]
The PAMAP2 Physical Activity Monitoring dataset contains data of 18 different physical activities, performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor.
### b. Augmentations
For the augmentations, we used the ones which were most effective as listed in [2]
#### Rotation
A method for simulating different sensor positions by plotting a uniformly distributed 3D random axis and a random rotation angle and applying the corresponding rotation to the sample
#### Scaling
Multiply by a random scalar to scale the size of the data in the window to simulate the motion of weaker magnitudes
#### Magnify
Multiply by a random scalar to magnify the size of the data in the window to simulate stronger amplitude motion
#### Resampling
Simulates multiple disturbances by varying the sampling frequency of sensor data
#### Noise Addition
A method for simulating additional sensor noise by multiplying the raw sample values with values that match uniform distribution 

### c. Weight Decay
Weight decay adds a penalty to the loss function, which has been well known to enhance model generalization and help prevent overfitting. Here we have added l1 and l2 norms of weights to our existing loss function with coefficient for l1 being 1e-5 and coefficient of l2 being 1e-4.

### d. Models
The study involves a model architecture of varying complexity  to understand the impact of regularization. We used a small model with an LSTM architecture model having 54,706 parameters, a medium sized model with Deep ConvLSTM architecture having 416,716 parameters. A larger model with the DeepConvLSTM architecture having 589,388 parameters is also included. All the models used in the study are below.



![LSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/media/Models/LSTM_Model_Small.png)

![ConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/media/Models/ConvLSTM_Medium.png)

![DeepConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/media/Models/DeepConvLSTM_Large.png)


## 4. Evaluation and Results

### UCI HAR
  #### LSTM
  ![LSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/main/docs/Evaluation_Results/UCI_HAR_LSTM.JPG)
#### ConvLSTM 
  ![ConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/main/data/Evaluation_Results/UCI_HAR/UCI_HAR_Conv_LSTM.JPG)
 #### DeepConvLSTM
  ![DeepConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/main/docs/Evaluation_Results/UCI_HAR_DeepConv_LSTM.JPG)

### USC HAD
 #### LSTM
  ![LSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/main/docs/Evaluation_Results/USC_HAD_LSTM.JPG)
#### ConvLSTM 
  ![ConvLSTM](https:/raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/Evaluation_Results/USC_HAD_Conv_LSTM.JPG )
#### DeepConvLSTM
  ![DeepConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/Evaluation_Results/USC_HAD_DeepConv_LSTM.JPG)


### PAMAP
 #### LSTM
  ![LSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/Evaluation_Results/PAMAP_LSTM.JPG)
#### ConvLSTM 
  ![ConvLSTM](https://raw.githubusercontent.com/shreyasrajesh0308/ece209as_project/blob/main/docs/Evaluation_Results/PAMAP_Conv_LSTM.JPG)


We see that we do notice the same trends in some cases with the HAR datasets eventhough none are as pronounced as what we saw with Imagenet. We see that adding these techiques for regulrization come with their own pitfalls. For example adding a rotations augmentation to the USC-HAD dataset with a moving up an elevator class introduces a bias which drops the accuracy of these classes eventhough the overall accuracy see's an improvement. A possible explanation for this could be the decision boundaries learnt for certain classes are not as robust as other classes leading to a drop in accuracy as a little variation is introduced in the dataset, but a larger study has to be performed to further explore these findings. 

## 5. Discussion and Conclusion

In this project, we studied the impact of data augmentation and weight decay on human activity recognition datasets. Our findings indicate the best regularization varies across the model architecture and datasets.  Yet a significant difference is not noticed in class accuracies in human activity recognition when compared to image classification [1]. We considered a subset of the commonly used model architectures and data augmentation techniques. It is interesting to see the impact of regularization on Opportunity[12] and WISDM[13] dataset. A possible direction as an extension would be to use a combination of augmentation techniques to compare against the baseline model. Another promising approach to HAR datasets is using contrastive learning which has shown to have more accuracy[2] as well as using generative models for augmentation.  

## 6. References

[1]  Balestriero, Randall & Bottou, Leon & LeCun, Yann. (2022). The Effects of Regularization and Data Augmentation are Class Dependent. 

[2] Wang, Jinqiang and Zhu, Tao and Gan, Jingyuan and Chen, Liming and Ning, Huansheng and Wan, Yaping. (2021). Sensor Data Augmentation with Resampling for Contrastive Learning in Human Activity Recognition

[3] E. De-La-Hoz-Franco, P. Ariza-Colpas, J. M. Quero and M. Espinilla, "Sensor-Based Datasets for Human Activity Recognition – A Systematic Review of Literature," in IEEE Access, vol. 6, pp. 59192-59210, 2018, doi: 10.1109/ACCESS.2018.2873502

[4] Wen, Qingsong et al. “Time Series Data Augmentation for Deep Learning: A Survey.” IJCAI (2021)

[5] Cui, Zhicheng and Chen, Wenlin and Chen, Yixin. (2016). Multi-Scale Convolutional Neural Networks for Time Series Classification 

[6] S. Mekruksavanich and A. Jitpattanakul, “LSTM Networks using Smartphone data for Sensor-based Human Activity Recognition in smart homes,” Sensors, vol. 21, no. 5, p. 1636, 2021

[7] A. Hoelzemann, N. Sorathiya and K. Van Laerhoven, "Data Augmentation Strategies for Human Activity Data Using Generative Adversarial Neural Networks," 2021 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops), 2021, pp. 8-13, doi: 10.1109/PerComWorkshops51409.2021.9431046.

[8] Fu, B., Kirchbuchner, F., & Kuijper, A. (2020). Data augmentation for time series: traditional vs generative models on capacitive proximity time series. Proceedings of the 13th ACM International Conference on Pervasive Technologies Related to Assistive Environments

[9] https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

[10] https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

[11]  https://sipi.usc.edu/had/

[12] UCI Machine Learning Repository: OPPORTUNITY Activity Recognition Data Set

[13] UCI Machine Learning Repository: WISDM Smartphone and Smartwatch Activity and Biometrics Dataset Data Set




