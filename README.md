# Wells Fargo Machine Learning Competition

## Method

![This is a alt text.](/flowchart.png "This is a sample image.")

First we mapped all of the labels of the categorical feature XC to numbers (A→1,B→2,C→3,D→4,E→5). This was done not only to investigate the correlation between XC and the other numerical features, but also to simplify the processes of feature selection and delivering inputs to our model.

We then sought out to eliminate any dependent features. After calculating the correlation matrix of the 31 feature variables, we found that no two features had a correlation coefficient below -0.1 or exceeding 0.1. Thus concluded that the 31 features were all independent with one another.

Without any dependent features, we then began to eliminate features using a linear epsilon support vector machine (SVM) and an algorithm called recursive feature elimination (RFE) (Guyon, 2002). An SVM is a regression tool used in binary classification that tries to separate two classes of data with a hyperplane. In our case the two classes are y=0 and y=1 where y is the target variable for each data point. The RFE algorithm ranks the importance of the features and then eliminates the least important ones based on the coefficient sizes of an SVM fitted to separate the two classes in our training data as best as it can. After a period of trial and error, we found training our model on the 16 most important features resulted in the highest F1 score.

For the actual model we trained on our data, we decided on an artificial neural network because of its ability to learn from hidden trends in the data and due to its history of effectiveness in industry and research. The model was trained with stochastic gradient descent learning using cross entropy loss as our cost function to optimize.

Neural networks are normally built with hidden layers, but after some experimentation we found that our model performed the best when it didn’t have any hidden layers. After a period of trial and error, we also found our learning algorithm to be most effective with a learning rate of 0.005 and a momentum of  0.9. In all cases our model trained over the same dataset for 100 epochs for higher accuracy. In most machine learning models, the training data needs to be divided into sub-batches at the cost of learning precision due to computational constraints, but in our case our model was efficient enough that this was unneeded.

We tested our model using a technique called two-fold cross validation, in which we randomly divided the labeled dataset into two groups. The advantage of two-fold cross validation over simply testing the model on the same set it was trained on is that cross validation allows us to avoid creating a model that is very effective on its training data yet generalizes poorly to new data. We trained a model on the first set and tested it on the second set, and vice versa. The two scores were taken together to obtain an average F1 score as our final result, which measures how well our model can adapt to datasets it has never seen before.

## Assumptions
We assumed that the 31 feature variables provided to us were all independent. This is justified by the low correlation coefficients found in the correlation matrix of these variables. In our process of feature selection we also assumed that the data was linear in order to use an SVM. We found strong evidence that the data was indeed linear from that fact that the F1 score increased significantly after eliminating features using a linear SVM, and also by the fact that our neural network performed very efficiently without hidden layers. In our trial and error period of tweaking the model’s parameters, we assumed the F1 scores we received from testing were indicative of the model’s actual strength. Training our model is a non-deterministic process since the random shuffling of training data can cause the model’s F1 score to fluctuate. To combat this we tested our model several times on the same parameters and took the average F1 score as the score we compared with scores obtained from different parameters.

## Novelties in our Approach
Our approach is novel in its combination of a support vector machine with a recursive feature algorithm used to eliminate feature variables rather than modifying them. Additionally, when faced with the challenge of integrating a categorical variable with numerical features we chose to map labels to numbers instead of sorting the data points based on their category and training the model separately on each group. This resulted in a simplified method with much faster training times. We also used cross validation instead of same set training/testing in order to reduce bias from overfitting and make sure our model can generalize well to new data sets.

## Model Strengths/Limitations
Our model is highly effective at predicting the target variable. On the full training set of 3000 points our model can correctly label 99.6% of them with an F1 score of 0.9935 out of 1. The F1 score for a binary classifier is a measure of its precision in classifying data points correctly in addition to its robustness in not overlooking data points. By comparison simply guessing the target variable randomly results in an F1 score of about 0.385. Even after testing with two-fold cross validation, giving our model less data to learn from and then evaluating it on data it hasn’t seen, we were able to obtain an F1 score of 0.984.
 
Our model is also very computationally efficient. It requires less than a minute to train itself over the labeled dataset 100 times. We were able to achieve this efficiency without relying on any of the usual tricks that sacrifice precision in the learning algorithm for speed (e.g. separating the training data into mini batches).

Our model is also highly adaptable to new data sets as evidenced by its high F1 score under cross validation. Our approach can be generalized to any situation requiring prediction of a binary target variable based on linear data. Feature selection can be done on non-linear data by using a kernel function in tandem with the SVM. The model’s fast training speed also means its parameters can be tweaked to fit a new dataset without sacrificing significant time. Additionally, our method of feature selection eliminates features entirely rather than mapping them to a smaller set of brand new features (e.g. principal component analysis). This allows users of our model to make meaningful conclusions about the importance of certain variables relative to the binary target they wish to predict. Recursively eliminating features also means our model requires fewer inputs to operate effectively, resulting in fewer costs and challenges from data collection. Finally, by preserving the original features our model offers a level of transparency that can be used to weed out potential biases in its behavior. A user will know exactly what factors our model deems important toward predicting the target.

A limitation of our model is that the time it takes to select features scales significantly with an increase in the total number of features. Thankfully, this process only needs to be performed once to rank the features. From there a user may tweak the number of features and test them as normal. Another limitation is that our approach becomes more complex with non-linear data since choosing an appropriate kernel function for a dataset must be done on a case-by-case basis and is very difficult. However, this isn’t a challenge specific to our method. Rather, it’s a challenge that many approaches to non-linear machine learning face.

## Real World Applications

Typical applications associated with binary classification include predicting medical conditions based on patient data and automating quality control to determine whether a product or system meets certain requirements. Binary classification can also be used by businesses to predict customer satisfaction. For example, a bank could predict whether a customer will leave or stay with the bank using that person’s account information and which demographics they fall into.

Our approach is practical for real world use due to its adaptability to different scenarios and datasets. It is also cost effective due to its computational efficiency and feature selection to reduce data collection costs. The transparency in its feature selection also opens the doors to human analysis and bias prevention measures.

## Sources

Chen, Pai-Hsuen, Fan, Rong-En, and Lin, Chih-Jen. "A Study on SMO-type Decomposition Methods for Support Vector Machines." IEEE Transactions on Neural Networks 17, no. 4 (2006): 893-908. Available: https://www.csie.ntu.edu.tw/~cjlin/papers/generalSMO.pdf [Accessed 8/7/2020]

Guyon, I., Weston, J., Barnhill, S. et al. “Gene Selection for Cancer Classification using 
Support Vector Machines.” Machine Learning vol. 46, pp. 389–422, January 2002. Available: https://doi.org/10.1023/A:1012487302797 [Accessed 8/2/2020]

Schmidhuber, Jürgen. "Deep Learning in Neural Networks: An Overview." Neural Networks vol. 61, pp. 85-117, 2015. Available: https://arxiv.org/abs/1404.7828 [Accessed 7/27/2020]

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. Available: https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html [Accessed 8/12/2020]
 

