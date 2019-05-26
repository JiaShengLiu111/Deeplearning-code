run_test：原始用于测试的文件。

run_test_calcTime：再进行模型测试的同时，计算模型的预测速度，思路是：在predict函数中返回程序执行时间，然后将所有batch的程序执行时间相加，即可得到
             模型预测所有样本的时间（这个时间除以细胞数量，即模型单位时间内预测细胞的个数） 

修改：修改cost和performance图片的保存方式，每次仅仅保存最新的图片即可。（节省空间，便于浏览）

换了一个MobileNetV1的网络结构：
参考网址：https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py
训练结果仍然不理想，具体请见：C:\Users\17335\Desktop\MobileNet-V1训练过程\第三次训练
网络结构没应该没有问题，那就有可能是优化器和学习率的问题。
计划尝试不同种类的优化器和学习率。
突然发现，keras版本的MobileNet和tensorflow在实现时，优化器确实不相同，那就说明，很大可能是优化器的作用导致结果不相同，并且网络结构中全连接层也有差别。
鉴于这个问题，做了如下改进：
    1、增加一层1024全连接层。
    2、使用不同的优化器进行实验。优化器同：http://lanproxy.biodwhu.cn:9079/tree/17_LJS/Code/landing/MobileNet/MobileNetV2_Keras_NoTL_Formal_Pang
由于学习率导致的问题？？