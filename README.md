Data Science Final WIT 2020
========

* __[Andrew Galvin](https://github.com/andrewgalvin)__

* __[AJ Salaris](https://github.com/Oracle331)__

* __[Endi Caushi](https://github.com/mrendi29)__

* __[Yuan Gao](https://github.com/yuanionrings)__

Introduction
========
This project is a Python-based facial recognition emotion analysis. It utilizes primarily Keras and Numpy to take in a photo of a face and return a mood. The program consists of two main files, 'Process.py' and 'Train.py'. The process file takes in the input file and converts it into a 48x48 image which is then checked against the model to formulate the expression prediction. The training file similarly converts all the training data into 48x48 images and runs them through the modeler, which is using a sequential model. 
This project was chosen as we all were interested in the growing use of biometrics in technology. The initial question or problem to solve is how do we go out about determining someone's mood based on their facial expression. Using data sets of faces with different facial expressions and the accompanying emotion, we can determine an approximate mood of a person in any given picture. 


Selection of Data
========
The machine learning model was trained using data sets found from __[Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)__. The data used was some of the most simple data available to us. We wanted to ensure the data we used was tested and accurate, so if there was any trouble getting the model to succeed, we could be certain it was not a result of the data. Once the model was successfully trained and returning somewhat expected results, it became more noticable the model was potentially over-fit. On occassion it would classify various photos entirely wrong, which was likely a result of facial features that weren't present in any of the test data. 

Methods
========
In order to solve our initial question of 

Results
========
and what happens when you take the meth


Discussion
========
The model included is likely over-fit as it struggles with faces of people of different ethnicities with different facial features. This could be improved with a larger dataset and different amounts of training according to the CLT.


Summary
========
thats that 

References
========
* Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.
