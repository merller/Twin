# Twin



This repository is dedicated to providing comprehensive instructions on a running-code demonstration for the 47th ICSE 2024 paper entitled "Scenario-Oriented Programming for IoT."


## Introduction



The Twin is a platform for IoT programming which consists of three layers, the interface layer, the connection layer and the exeuction layer. 


## Getting started



### Environment requirements

* Python 3.12
* Java 21.04
* PyTorch 2.3.0
* Cuda 12.1


### How to run

* #### Twin
We provide a JAR package (Twin.jar) for compile scenario program. It can be exeucted using the following instruction:

```java -jar Twin.jar --programFilePath```

For example: ```java -jar Twin.jar Twin_jar/program.txt ``` 

* #### Connection layer
We specifically provide three demos for the connection layer, corresponding to the cluster module, the cluster determination module, and the code search module.

Go to the `demos` folder, download the folders `Models` and `dataSet-demo`. And then you can choose to donwload which one scipt or all scripts for demos of the above three modules. 


