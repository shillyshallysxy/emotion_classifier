# emotion_classifier
emotion classifier based on kaggle fer2013

基于Keras框架搭建并训练了卷积神经网络模型，用于人脸表情识别，训练集和测试集均采用kaggle的fer2013数据集
达到如下效果：

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample3.png)

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample2.png)

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample4.png)

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample5.png)

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample6.png)

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/sample1.png)

最后因为更换电脑（升级了1080ti）原模型丢失，随手用以上的文件重新训练了一个模型，没有做什么调整优化

以下是训练好的模型在test集上的混淆矩阵，以及在整个test上的准确度

![image](https://github.com/shillyshallysxy/emotion_classifier/blob/master/pic/acc.png)
