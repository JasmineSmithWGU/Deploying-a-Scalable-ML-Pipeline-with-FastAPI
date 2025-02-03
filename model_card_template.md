# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Machine Learning Adult income prediction model based on income that exceeds $50K/yr. This model was created using Census Data from 1994 from ages 16-100 and formed through Scikit-learn. Through modeling, it is designed to predict whether an individuals demographic or employment data effect their income exceeding $50k per year. RandomForestClassifer is used to categorize individuals and features demographic data such as age, education level, marital status, occupations, and hours worked per week.
Link: (https://archive.ics.uci.edu/dataset/20/census+income)

## Intended Use
This model is intended to predict individual cliassifiactions based on income. Identifying the factors on whether they earn more than $50k or less. This type of model can be used for market analysis and research. 
## Training Data
Dataset: Census Income 
Data Source: (https://archive.ics.uci.edu/dataset/20/census+income)
## Evaluation Data
The test consists of 8000+ samples and inlcludes individuals from different backgrounds. The evaluation conducted used the same features used during training. 
Categorical features that were reviewed: workclass, education, marital-status, occupation, relationship, race, sex, and native-country.
## Metrics
Precision: 0.7230
Recall: 0.6100
F1 Score: 0.6617

The results show that the model works well to define the individuals earning more than $50K with a midlevel balance between recall and precision. 
## Ethical Considerations
There are a few considerations to include. Potential bias from an imbalanced dataset. There could be less representatoin for lower class income earner making the model leader more favorably towards the higher income based on occupation and educational background. Also, the data that is provided is from a different period of time that doesn't accurately represent current income earners. Occupations can be scued as well as the types of jobs shown may not clearly represent current job markets. 
## Caveats and Recommendations
To improve the model, more data would be required to represent each group equally. 
