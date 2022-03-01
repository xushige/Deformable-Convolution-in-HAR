# [IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT 2022]Deformable-Convolution-for-HAR
[IEEE TIM 2022] Deformable Convolutional Networks for Multimodal Human Activity Recognition using Wearable Sensors
![Model](https://github.com/wenbohuang1002/-IEEE-JBHI-2021-Channel-Selectivity-CNN-for-HAR/blob/main/Images/Model.png)
Here shows the simplfied TRAIN process on benchmark public datasets.
Thanks for pointing out improper!

## Requirements in this work
● Python 3.8.10  
● PyTorch 1.8.2 + cu111
● Numpy 1.21.2

## Train
Get required dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets
```
$ cd Deformable-Convolution-in-HAR
$ python main.py
```

