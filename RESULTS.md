# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

- Which metric performed best and why?
- Which metric performed worst and why?
- How much did the class imbalance affect the results?
- What does the confusion matrix tell you about the model's predictions?

*The metric that performed the best was AUC, which was 0.8853. Accuracy was 0.9195, which is indicative of overall good performance, but the accuracy metric isn't that great with identifying the positive cases (recall was low, at 0.3239). The precision was also relatively low, at 0.6765. With these considered, althought there was accurate identification and a relatively high AUC, the imbalance caused the precision and recall to be lower and missed a lot of the positive cases. The overall imbalance impact score was approximately 0.3525. When observing the confusion matrix in the results_part1.txt file, the matrix supports this analysis, with many of the cases correctly identified, but a larger proportion of the positive cases being miscategorized.*

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

- Which model performed better according to AUC score?
- Why might one model outperform the other on this dataset?
- How did the addition of time-series features (rolling mean and standard deviation) affect model performance?

*The Random Forest AUC was 0.9735 while the XGBoost AUC was 0.9953. While both are relatively high, the XGBoost Model was slightly higher than the Random Forest Model. It's possible this was due how the gradient boost was applied. When comparing the rolling mean and SD values, it appears that it was more successful in improving the model performance than without.*

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
- Which metrics showed the least improvement?
- Why might some metrics improve more than others?
- What does this tell you about the importance of addressing class imbalance?

*The metrics that showed the most significant improvement was recall rate, which was better by approximately 152%. The next was f1 and AUC, which also showed slight improvment. The metrics which had less improvment was precision (decreased by almost 49%) and accuracy (decreased by almost 9.5%). This is likely the case due to using SMOTE, which works to distribute some of the imbalance (we see that improvement in recall). However precision dropped as a result, which suggests there are a lot of false positives still being identified. When looking at the confusion matrix, we see that that is the case. It tells us that even though some changes in the processing can address the class imbalance, it can come at the cost of decreasing another metric. I think that in the case of this data, having a higher recall and pinpointing positives would be better than precision or recall.*

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

*Some of the important factors affecting model performance would include understanding possible imbalance between the data and how to address it (like using SMOTE), looking at the different data models and AUC metrics, and looking at data over time (like when we added rolling values). The rolling values, XGBoost, and SMOTE all showed some aspect of improvement in the data modeling. For future modeling, I would investigate more into the imbalance and possible alternatives to address it, and relooking at how to minimize the amount of false positives in the data to improve both recall and precision.*