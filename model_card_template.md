# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier trained to predict whether an individual earns <=50K or >50K annually based on U.S. Census Bureau demographic and employment data.
The model uses:
-One-hot encoding for categorical features
-A label binarizer for the target variable
-A fixed random seed (random_state=42) for reproducibility
-100 decision trees (default RandomForest configuration)

The model and encoder are saved as serialized .pkl files and loaded during inference through a 
FastAPI application.

## Intended Use
The model is intended for educational purposes within the Udacity Machine Learning DevOps Engineer
Nanodegree.
Its primary purpose is to demonstrate:

-Building a machine learning pipeline
-Training and evaluating a classification model
-Computing performance on data slices
-Deploying a model using FastAPI
-Writing unit tests and CI/CD workflows

This model should not be used for real-world decision-making, hiring, creadit scoring or any
application that affects individuals' rights or opportunites.

## Training Data
The model was trained on the Cencus Income dataset, a publicly available dataset containing 
demographic and employment attributes such as:

-Age
-Workclss
-Education
-Marital status
-Occupation
-Relationship
-Race
-Sex
-Hours per week
-Native country 
-Capital gain/loss

The dataset included a binary income label (salary) indicating whether a person earns <=50K or >50K.

A train/test split of 80/20 was used. Categorical features were one-hot encoded, and continuous features were left unscaled.

## Evaluation Data
The evaluation dataset is the 20% test split from the original Census Income dataset. 
The same preprocessing steps were applied:

-One-hot encoding using the trained encoder
-Label binarization using the trained label binarizer

Model performance was computed on:
-The full test set
-Invidual slices of categorical features (e.g., education, sec, race, workclass)

## Metrics
The model was evaluated using:
-Prediction
-Recall
-F1 score (F-beta with ß=1)

These metrics were chosen because the dataset is imbalanced and classification quality must be 
measure beying accuracy.

Your actual metrics (from your train_model.py output) should be pasted here.
For example:
Precision: 0.78
Recall:    0.63
F1 Score:  0.70

Slice metrics (per categorical value) are included in slice_output.txt

## Ethical Considerations
The Census dataset contains sensitive demographic attributes such as race, sex, and marital status.
Using these features in predictive models can introduce or amplify biasm especially when:

-The model is used for decision-making addecting individuals
-The dataset reflects historical inequalities
-Certain demographic groups are under-represented

RandomForest model do not inherently mitigate bias, and one-hot encoding preserves all categorical distinctions. Therefore, predictions may cary across demographic slices, and these differences must be carefully evaluate before any real-world use. 

This model must not be used for employment, financial, housing, or legal decisions. 

## Caveats and Recommendations
-The model is trained on historical Census data and may not generalize to modern populations.
-The dataset contains noise, missing values, and potential reporting errors,
-The model does not perform feature scaling, which may affect performance for certain algorithms.
-The model does include hyperparameter tuning; performance could be improved with grid search or randomized search.
-The model should be retrained periodically is used in any real application.
-Slice performance should be monitored continuously to detect bias or degration.
