# Deformable-Convolution-for-HAR
## JOURNAL: [IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT 2022] 
## TITLE: [Deformable Convolutional Networks for Multimodal Human Activity Recognition using Wearable Sensors]
<div align="center">
  <img src="model_fig/modelbig.png" width="1050"/>
</div>
<br />
### Here shows the simplfied TRAIN process on benchmark public datasets. Thanks for pointing out improper!
### Requirements in this work
1. Python 3.8.10  
2. PyTorch 1.8.2 + cu111
3. Numpy 1.21.2
### Train
Get required dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets
```
$ cd Deformable-Convolution-in-HAR
$ python main.py
```
