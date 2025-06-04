# Machine Learning

This directory contains tutorials and implementations of various machine learning algorithms and techniques.

## Contents

### Tutorials and Implementations
This directory contains a comprehensive collection of tutorials and implementations for various machine learning algorithms, techniques, and functions. Each topic is typically presented in a Jupyter notebook format and includes the following:

-   **Theoretical Background:** Explanation of the core concepts and principles.
-   **Mathematical Formulations:** Detailed equations and derivations.
-   **Implementation from Scratch:** Building the algorithm using fundamental libraries (e.g., NumPy).
-   **Implementation using Libraries:** Utilizing popular machine learning libraries (e.g., scikit-learn).
-   **Visualizations:** Graphs and plots to understand data and model behavior.
-   **Result Analysis:** Interpretation of model performance and outcomes.
-   **Practical Applications/Exercises:** Examples and tasks to solidify understanding.

The topics covered include:

1.  **[Linear Regression](tutorials/01_Linear_Regression/)**:
    -   [Simple Linear Regression](tutorials/01_Linear_Regression/01_Simple_Linear_Regression/)
    -   [Multiple Linear Regression](tutorials/01_Linear_Regression/02_Multiple_Regression/)
    -   [Polynomial Regression](tutorials/01_Linear_Regression/03_Polynomial_Regression/)

2.  **[Regularization and Regression](tutorials/02_Regularization_And_Regression/)**:
    -   Techniques to prevent overfitting in regression models, such as Ridge, Lasso, and Elastic Net.

3.  **[Logistic Regression](tutorials/03_Logistic_Regression/)**:
    -   [Binary Logistic Regression](tutorials/03_Logistic_Regression/01_Logistic_Regression/)
    -   [Multinomial Logistic Regression](tutorials/03_Logistic_Regression/02_Multinomial_Logistic_Regression/)
    -   [Softmax Regression](tutorials/03_Logistic_Regression/03_Softmax_Regression/)

4.  **[Naïve Bayes](tutorials/04_Naïve_Bayes/)**:
    -   [Introduction to Naive Bayes](tutorials/04_Naïve_Bayes/01_Introduction_To_Naive_Bayes/)
    -   [Gaussian Naive Bayes](tutorials/04_Naïve_Bayes/02_Gaussian_Naive_Bayes/)
    -   [Multinomial Naive Bayes](tutorials/04_Naïve_Bayes/03_Multinomial_Naive_Bayes/)
    -   [Bernoulli Naive Bayes](tutorials/04_Naïve_Bayes/04_Bernoulli_Naive_Bayes/)
    -   [Complement Naive Bayes](tutorials/04_Naïve_Bayes/05_Complement_Naive_Bayes/)

5.  **[Decision Tree](tutorials/05_Decision_Tree/)**:
    -   Implementation and understanding of decision tree algorithms for classification and regression.

6.  **[Random Forest](tutorials/06_Random_Forest/)**:
    -   Ensemble method utilizing multiple decision trees for improved performance.

7.  **[Support Vector Machine (SVM)](tutorials/07_Support_Vector_Machine/)**:
    -   Implementation of SVM for classification and regression tasks, including different kernels.

8.  **[K-Nearest Neighbors (KNN)](tutorials/08_K_Nearest_Neighbors/)**:
    -   A simple, instance-based learning algorithm used for classification and regression.

9.  **[Ensemble Learning](tutorials/09_Ensemble_Learning/)**:
    -   Techniques combining multiple models to improve predictive performance:
        -   [Bagging](tutorials/09_Ensemble_Learning/01_Bagging/)
        -   [Boosting](tutorials/09_Ensemble_Learning/02_Boosting/)
        -   [Voting](tutorials/09_Ensemble_Learning/03_Voting/)
        -   [Stacking](tutorials/09_Ensemble_Learning/04_Stacking/)

10. **[K-Means Clustering](tutorials/10_K_Means_Clustering/)**:
    -   An unsupervised learning algorithm for partitioning data into clusters.

11. **[Principal Component Analysis (PCA)](tutorials/11_PCA/)**:
    -   A dimensionality reduction technique.

12. **[Loss Functions](tutorials/12_Loss_Function/)**:
    -   Common functions used to measure the error of a model:
        -   [Mean Absolute Error (MAE)](tutorials/12_Loss_Function/01_Mean_Absolute_Error/)
        -   [Mean Squared Error (MSE)](tutorials/12_Loss_Function/02_Mean_Squared_Error/)
        -   [Cross-Entropy Loss](tutorials/12_Loss_Function/03_Cross_Entropy_Loss/)

13. **[Evaluation Functions](tutorials/13_Evaluation_Function/)**:
    -   Metrics used to assess the performance of machine learning models:
        -   [Accuracy](tutorials/13_Evaluation_Function/01_Accuracy/)
        -   [Precision](tutorials/13_Evaluation_Function/02_Precision/)
        -   [Recall](tutorials/13_Evaluation_Function/03_Recall/)
        -   [F1 Score](tutorials/13_Evaluation_Function/04_F1_Score/)

14. **[Optimization Functions (Optimizers)](tutorials/14_Optimization_Function/)**:
    -   Algorithms used to minimize the loss function:
        -   [Gradient Descent](tutorials/14_Optimization_Function/01_Gradient_Descent/)
        -   [Stochastic Gradient Descent (SGD)](tutorials/14_Optimization_Function/02_Stochastic_Gradient_Descent/)
        -   [L-BFGS](tutorials/14_Optimization_Function/03_L-BFGS/)
        -   [Momentum](tutorials/14_Optimization_Function/04_Momentum/)
        -   [Nesterov Accelerated Gradient (NAG)](tutorials/14_Optimization_Function/05_Nesterov_Accelerated_Gradient/)

15. **[Activation Functions](tutorials/15_Activation_Function/)**:
    -   Functions used in neural networks to introduce non-linearity:
        -   [Sigmoid Function](tutorials/15_Activation_Function/01_Sigmoid_Function/)
        -   [Softmax Function](tutorials/15_Activation_Function/02_Softmax_Function/)
        -   [ReLU (Rectified Linear Unit) Function](tutorials/15_Activation_Function/03_ReLU_Function/)
        -   [Leaky ReLU Function](tutorials/15_Activation_Function/04_LeakyReLU_Function/)

16. **[Distance Functions](tutorials/16_Distance_Function/)**:
    -   Metrics used to calculate the distance or similarity between data points:
        -   [Euclidean Distance](tutorials/16_Distance_Function/01_Euclidean_Distance/)
        -   [Cosine Similarity](tutorials/16_Distance_Function/02_Cosine_Similarity/)

17. **[Embedding Functions](tutorials/17_Embedding_Function/)**:
    -   Techniques for representing data in a lower-dimensional space:
        -   [Autoencoders](tutorials/17_Embedding_Function/Autoencoders_Function/)
        -   [GloVe](tutorials/17_Embedding_Function/GloVe_Function/)
        -   [word2vec](tutorials/17_Embedding_Function/word2vec_Function/)

18. **[Kernel Functions](tutorials/18_Kernel_Function/)**:
    -   Functions used in algorithms like SVM to map data into higher-dimensional spaces:
        -   [Linear Kernel](tutorials/18_Kernel_Function/01_Linear_Kernel/)
        -   [Polynomial Kernel](tutorials/18_Kernel_Function/02_Polynomial_Kernel/)
        -   [RBF (Radial Basis Function) Kernel](tutorials/18_Kernel_Function/03_RBF_Kernel/)

### Structure

Each tutorial typically includes:
- Theoretical background
- Mathematical formulations
- Implementation from scratch
- Implementation using scikit-learn
- Visualizations and result analysis
- Practical applications and exercises

## Requirements

The following Python libraries are required to run the tutorials:
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- seaborn
- jupyter

Install all requirements by running:
```
pip install -r requirements.txt
```

## Usage

1. Make sure all requirements are installed
2. Launch Jupyter Notebook in this directory:
   ```
   jupyter notebook
   ```
3. Open the desired tutorial notebook
4. Execute cells sequentially to follow along with the lessons

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.