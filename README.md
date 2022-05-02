# Credit_Risk_Analysis

Jill asked me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. LendingClub gave me a credit card dataset. I will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then use the SMOTEENN algorithm a combination of over and under sampling.

Then I will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Resources

- Data Source: LoanStats_2019Q1.csv
- Jupyter Lab 6.4.8
- Python 3.7.11

## Results

### Naive Random Oversampling

![image](https://user-images.githubusercontent.com/96445453/166183039-ea0ccde2-22a1-4e59-a9e0-9184d7b8e2e5.png)
![image](https://user-images.githubusercontent.com/96445453/166183436-7b6de6d6-9017-4724-baf7-36f733b855c3.png)

- This model's balanced accuracy score is 66.2%
- There are a large number of low_risk population, with a sensitivity of 60% and precision of 100%
- The precision of the high_risk population is 1% with a sensitivity of 72%, our F1 is 2%

### SMOTE Oversampling

![image](https://user-images.githubusercontent.com/96445453/166183733-dee29329-72bc-446f-ac5c-601eafb65dd6.png)
![image](https://user-images.githubusercontent.com/96445453/166183764-9e68d306-1765-43c8-8ef7-a5b633343773.png)

- This model's balanced accuracy score is 65.68%
- There are a large number of low_risk population, this model has a sensitivity of 70% and precision of 100%
- The precision of the high_risk population is 1% with a sensitivity of 61%, our F1 is 2%

### Undersampling ClusterCentroids

![image](https://user-images.githubusercontent.com/96445453/166184056-7cb91d49-a7de-438f-9c08-ca84386a67ee.png)
![image](https://user-images.githubusercontent.com/96445453/166184064-77016326-cab7-4cb6-b7b8-383abf47e80b.png)

- This model's balanced accuracy score is 54.47%
- There are a large number of low_risk population, this model has a sensitivity of 40% (due to the large amount of false positives) and precision of 100%
- The precision of the high_risk population is 1% with a sensitivity of 69%, our F1 is only 1%

### Combination (Over and Under) Sampling, SMOTEENN

![image](https://user-images.githubusercontent.com/96445453/166184344-f02027a0-bc79-4b17-a3c4-ff0ac92b080b.png)
![image](https://user-images.githubusercontent.com/96445453/166184362-0e1c2ebb-71e8-42da-b57a-d5edceb47462.png)

- This model's balanced accuracy score is 64.61%
- There are a large number of low_risk population, this model has a sensitivity of 57% and precision of 100%
- The precision of the high_risk population is 1% with a sensitivity of 72%, our F1 is 2%

### Balanced Random Forest Classifier

![image](https://user-images.githubusercontent.com/96445453/166184617-dfe3b517-d273-47ce-af17-c7363270c907.png)
![image](https://user-images.githubusercontent.com/96445453/166184627-65b2687a-89c5-4168-9d53-398f805aea50.png)

- This model's balanced accuracy score is 78.85%
- There are a large number of Predicted low_risk population, this model has a sensitivity of 87% and precision of 100%
- The precision of the high_risk population is 3% with a sensitivity of 70%, our F1 is 6%

### Easy Ensemble AdaBoost Classifier

![image](https://user-images.githubusercontent.com/96445453/166184844-ae6c5700-77af-43d2-999c-ad03af2cc881.png)
![image](https://user-images.githubusercontent.com/96445453/166184861-70975253-c9d3-4c6d-a01c-dfef2e20ef77.png)

- This model's balanced accuracy score is 93.17%
- There are a large number of Predicted low_risk population, this model has a sensitivity of 94% and precision of 100%
- The precision of the high_risk population is 9% with a sensitivity of 92%, our F1 is 16%

## Summary

Based on the results of running all the models on the LendingClub credit card dataset we can determine a number of things.
- The precision of high_risk population is very low and a poor indicator to determine if credit risk is high
- The best model to represent the dataset is the Easy Ensemble AdaBoost Classifier. It had a balanced accuracy score of over 93%
- There are a lot of false positives in every model for actual low_risk combined with the high_risk having low precision more data is necessary to accurately predict Actual high_risk credit

I recommend the LendingClub use the Easy Ensemble AdaBoost Classifier model as it is the best fit for this dataset.
