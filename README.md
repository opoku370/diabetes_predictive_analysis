# PREDICTIVE ANALYSIS FROM BASIC HEALTHCARE DATA : A MACHINE LEARNING APPROACH

## Table of contents 

- [Objective](#objective)
- [Backgroung](#background)
- [Data Source](#data-source)
- [Dataset Summary](#dataset-summary)
- [Tools and Technologies](#tools-and-technology)
- [Project Workflow](#project-workflow)
  - [Data Exploration](#data-exploration)
    - [Loading the Data](#loading-the-data)
    - [Duplicate Check](#duplicate-check)
    - [Data Overview](#data-overview)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Outlier Check](#outlier-check)
    - [Analyzing Categorical Features](#analyzing-categorical-features)
  - [Data Engineering](#data-engineering)
    - [Feature Creation](#feature-creation)
    - [Feature Selection](#feature-selection)
    - [Feature Encoding of Categorical Variables](#feature-encoding-of-categorical-variables)
    - [Feature scaling](#feature-scaling)
  - [Model Selection and Training](#model-selection-and-training)
    - [Lazy Predict](lazy-predict)
    - [Support Vector Machine](support-vector-machine)
    - [Decision Tree](decision-tree)
    - [K-Nearest Neighbors](k-nearest-neighbors)
  - [Model Evaluation](model-evaluation)
    - [Support Vector Machine](support-vector-machine)
    - [Decision Tree](decision-tree)
    - [K-Nearest Neighbors](k-nearest-neighbors)
  - [Model Tuning](#model-tuning)
    - [Grid Search](#grid-search)
    - [Random Search](#random-search)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)
- [Impact](#impact)

## OBJECTIVE 
The primary objective of this project is to develop a predictive model that classifies individuals as diabetic or non-diabetic based on readily available health features such as age, gender, blood sugar level, weight, and height. By leveraging machine learning algorithms, the goal is to assist healthcare professionals or early screening platforms in identifying individuals at risk of diabetes for timely intervention.

## BACKGROUND
Diabetes is one of the most common chronic illnesses globally and is often undiagnosed in its early stages. Early prediction and management of diabetes can drastically reduce health complications and associated medical costs. Traditional diagnostic processes often require invasive or lab-based testing, whereas this project explores the feasibility of using non-invasive, basic features to predict diabetes risk.

## DATA SOURCE 
Data collected from a hospital in Ghana 
-<a href= “assets/dataset/diabetes.xlsx”> Dataset </a>

## DATASET SUMMARY
The dataset used in this project contains 625 records, each representing an individual's health profile with the following attributes:
-	Age: Patient's age in years
-	Gender: Categorical variable (Male/Female)
-	Sugar Level: Measured blood glucose level (mg/dL)
-	Weight: Body weight in kilograms
-	Height: Height in centimeters
-	Diabetes: Target variable indicating diabetes status (Yes/No)

## TOOLS AND TECHNOLOGIES
-  Python (core language)
-  Pandas, NumPy (data handling)
-  Scikit-learn (modeling, evaluation, preprocessing)
-  Jupyter Notebooks (experimentation and reporting)


## PROJECT WORKFLOW
### Data Exploration
The act of exploring your data is known as exploratory data analysis, and it usually include looking at the dataset's components and structure, the distributions of its individual variables, and the connections between two or more variables.

#### Loading the data
This code brings to life the data into the environment and the necessary libraries needed to complete analysis of the data then checking the count of total columns and rows, the data types of the various columns and null. 

``` Python

#Loading the libraries

import pandas as pd

#Loading the data

diabetes_pre= pd.read_csv(r"C:\Users\PC\Desktop\learn\dataset\diabetes.csv") 

#Dropping the fullname column 

diabetes_pre= diabetes_pre.drop(['fullname'],axis = 1)

#Assigning index to the data

diabetes_pre['Id'] = [f'{i:04}' for i in range(1, len(diabetes_pre) + 1)]

#renaming age

diabetes_pre.rename(columns={'age':'Age'},inplace = True)

#Looking at the data

diabetes_pre.head()

```

#### Output 
![Loading the data](assets/images/loaded%20data%20.png)


#### Duplicate Check 

```
#Checking for duplicates in the data

duplicates_pre= diabetes_pre[diabetes_pre.duplicated()]

```

#### Output 
![Duplicate Check](assets/images/duplicate%20check.png)



#### Overview of data

```
# Overview of the data( checking for null, and the datatpes) 

diabetes_pre.info()
```

#### Output
![Overview of Data](assets/images/overview%20of%20data.png)



#### Descriptive Statistics

```
#Descriptive statistics on the numerical features

diabetes_pre.describe()

```

#### Output 
![Descriptive Statistics](assets/images/descriptive%20statistics.png)



#### Looking for Outlier

```
# Columns to check

cols_to_check = ['Age', 'Sugar_Level', 'Weight', 'Height']

for col in cols_to_check:
    Q1 = diabetes_pre[col].quantile(0.25)
    Q3 = diabetes_pre[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = diabetes_pre[(diabetes_pre[col] < lower_bound) | (diabetes_pre[col] > upper_bound)]
    
    print(f"\nColumn: {col}")
    print(f"Total outliers: {len(outliers)}")
    print(outliers[[col]])
```

#### Output

![Looking for Outlier](assets/images/outlier%20check.png)



#### Analyzing Categorical Features

```
object_cols= ['Gender','Diabetes']

#finding unique values for the object columns

unique_values = {}
for col in object_cols :
    unique_values[col] = diabetes_pre[col].unique()

# printing the values for object_cols
for col, values in unique_values.items() :
    print(f"Unique values in '{col}' : ")
    print(values)
    print()

```

#### Output
![Analyzing Categorical Features](assets/images/unique%20values%20for%20object%20columns.png)


The object columns contains the correct unique records
Duplicate check was conducted with no duplicate in the data, for the overview of the data, each column was assigned the right data type with zero non-null count for the columns. For the descriptive statistics on the numerical features;
-	The dataset includes 625 records with complete data for age, Sugar_Level, Weight, and Height.
-	Age ranges from 1 to 40 years, with a mean of 21.15 years, indicating a young population that includes both children and adults.
-	Sugar levels vary widely (70 to 199 mg/dL), with a mean of 134.21 mg/dL, suggesting that some individuals may be at risk of diabetes.
-	Weight ranges from 50 to 99.97 kg, averaging 75.37 kg, showing moderate variability across individuals. -Height spans from 150 to 199.85 cm, with a mean of 174.34 cm, indicating a diverse height range, possibly across different age and gender groups.
Using the Outlier detection check for the numerical columns, All Age, Sugar_Level, Weight and Height entries fall within the expected range of variation. For the objects columns contains the correct unique records of Male and Female for Gender and No and Yes for the Diabetes target variable.


### Data Engineering
Which includes feature creation, feature selection, feature encoding and feature encoding is a pre-processing step which helps to improve model performance, reduce overfitting and ensure fair treatment of all input variables during training.

#### Feature Creation
```
# Height is in cm and Weight in kg

diabetes_pre['BMI'] = diabetes_pre['Weight'] / ((diabetes_pre['Height'] / 100) ** 2)
def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

diabetes_pre['BMI_Category'] = diabetes_pre['BMI'].apply(classify_bmi)

```

Using feature engineering involves adding new features to a model's feature vector that are calculated from the existing features. These new properties could be mathematical transformations of pre-existing traits, such as ratios or differences. The Added feature is BMI(Body Mass Index) which estimates body fat based on a person's weight and height. Calculated Using the formula: BMI= Weight(kg)/ [Height(m)]2. BMI greater or equal to 25 is associated with overweight or obesity which is a major risk factor for type 2 diabetes.

#### Feature Selection
All the available features appear to be relevant for predicting diabetes. Given the small number of input variables, the risk of overfitting is relatively low. Therefore, applying feature selection techniques such as filter methods (e.g., correlation analysis, F-test, Chi-square test), wrapper methods, or embedded methods is not considered necessary for this project.


#### Feature encoding of categorical variables 

```
#Making a copy of diabetes_pre as diabetes_eng 
diabetes_eng = diabetes_pre.copy()

#Encoding Categorical variables

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Encode Gender and Diabetes Column
diabetes_eng['Gender'] = diabetes_eng['Gender'].map({'Male': 0, 'Female': 1}) 
diabetes_eng['Diabetes'] = diabetes_eng['Diabetes'].map({'No': 0, 'Yes': 1})

# Encode BMI Category
le = LabelEncoder()
diabetes_eng['BMI_Category'] = le.fit_transform(diabetes_eng['BMI_Category'])   

```

#### Feature Scaling

```
from sklearn.preprocessing import StandardScaler

# List of columns to scale
columns_to_scale = ['Age', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'Gender' , 'BMI_Category']

# Initialize the scaler
scaler = StandardScaler()

# Apply StandardScaler
df_scaled_values = scaler.fit_transform(diabetes_eng[columns_to_scale])

# Convert back to DataFrame for easy use
diabetes_scaled = pd.DataFrame(df_scaled_values, columns=columns_to_scale)


# View scaled data
print(diabetes_scaled.head())

```

#### Output
![Feature Scaling](assets/images/feature%20scaling%20.png)


Feature scaling is an important step in machine learning as it standardizes the range of numerical features, leading to improved model performance and faster convergence during training. Without scaling, features like Height and Sugar Level, which have larger numerical ranges, could disproportionately influence the model, even if they are not inherently more important. While scaling is generally not required for tree-based models (e.g., Decision Trees, Random Forests), applying it does not negatively impact their performance and can still be beneficial when using a mix of algorithms.


### Model Selection and Training
#### Lazy Predict
```
!pip install lazypredict

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']  # 0 or 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

```

#### Output
![Lazy Predict](assets/images/lazy%20predict%201.png) 
![Lazy Predict](assets/images/lazy%20predict%202.png)


After evaluating model performance using LazyPredict, which provides a quick comparison of multiple algorithms, I selected Support Vector Classifier (SVC), Decision Tree, and K-Nearest Neighbors (KNN) for further analysis based on their performance metrics.

#### Support Vector Machine

```
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Define features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']  

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
 
```

#### Output

![Support Vector Machine](assets/images/svc%20model%20training.png)



#### Decision Tree

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Define features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')  
dt_model.fit(X_train, y_train)

```

#### Output
![Decision Tree](assets/images/decision%20tree%20model%20training%20.png)



#### K-NEAREST NEIGHBORS

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

```

#### Output
![K-Nearest Neighbors](assets/images/knn%20model%20training%20.png)



### Model Evaluation

#### Support Vector Machine

```
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on test data
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("Accuracy Score:", accuracy_score(y_test, y_pred_svm))

```

#### Output
![Support Vector Machine](assets/images/model%20evaluation%20KNN.png)



#### Decision Tree

```
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
print("Accuracy Score:", accuracy_score(y_test, y_pred_dt))

```

#### Output
![Decision Tree](assets/images/model%20evaluation%20decision%20tree.png)



#### K-Nearest Neighbors

```

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on test data
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("Accuracy Score:", accuracy_score(y_test, y_pred_knn))

```

#### Output
![K-Nearest Neighbors](assets/images/model%20evaluation%20KNN.png)


To predict diabetes diagnosis (Yes/No), three different classification models were evaluated: Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbors (KNN). The models were assessed using the following metrics:
 - Accuracy: Overall correctness of the model.
 - Precision: Correctness of positive predictions (Yes/No separately).
 - Recall: Ability to detect all actual positives.
 - F1 Score: Harmonic mean of precision and recall, providing a balance between the two.
 - Class Labels: 0 = No Diabetes, 1 = Yes Diabetes

Based on the classification reports, Support Vector Machine (SVM) is selected as the best-performing model for this project. It offers the highest recall and F1-score for the diabetic class, which is critical in healthcare-related applications where false negatives (missing a diabetic patient) must be minimized.
While KNN had more balanced metrics across classes, SVM’s high recall for detecting diabetes gives it an edge for real-world use where the cost of missing positive cases is high.


### Model Tuning for SVC
To enhance the performance of the initial Support Vector Machine (SVM) model, hyperparameter tuning was carried out using two methods: Grid Search and Randomized Search. The goal was to identify the best configuration of SVM hyperparameters (C, gamma, and kernel) to improve model accuracy and class-wise performance.


#### Grid Search SVM

```
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']  

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']  # Only used for rbf, poly, sigmoid
}

# Initialize base SVC model
svc = SVC(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    svc, param_grid=param_grid,
    cv=5, n_jobs=-1, verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_

# Predict on test data
y_pred_svm = best_svm.predict(X_test)

# Evaluate the model
print("Best Parameters Found:", grid_search.best_params_)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("Accuracy Score:", accuracy_score(y_test, y_pred_svm))

```

#### Output
![Gridsearch](assets/images/gridsearch%20svc.png)



#### Randomized Search SVM

```
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import uniform

# Define features and target
X = diabetes_scaled[['Age', 'Gender', 'Sugar_Level', 'Weight', 'Height', 'BMI', 'BMI_Category']]
y = diabetes_eng['Diabetes']  

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define hyperparameter distribution
param_dist = {
    'C': uniform(0.1, 10),  # C from 0.1 to 10
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']  # only relevant for non-linear kernels
}

# Initialize base SVC model
svc = SVC(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    svc, param_distributions=param_dist,
    n_iter=20, cv=5, random_state=42, n_jobs=-1
)

# Fit model
random_search.fit(X_train, y_train)

# Best model
best_svm = random_search.best_estimator_

# Predict on test data
y_pred_svm = best_svm.predict(X_test)

# Evaluate the model
print("Best Parameters Found:", random_search.best_params_)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("Accuracy Score:", accuracy_score(y_test, y_pred_svm))
```

![Random Search](assets/images/randomized%20search%20svc.png)

- Grid Search SVM maximized recall for diabetic patients (class 1), predicting all correctly, but at the cost of completely misclassifying non-diabetics.
- Randomized Search SVM offered a more balanced performance, but didn’t outperform the baseline model in any key metric.
 - The Baseline SVM remains the most reliable and balanced choice, particularly for medical prediction tasks where correctly identifying diabetic patients is crucial.
With the small size of the data, it does not do so well with hyperparameter tunning causing overfitting.


## RECOMMENDATION
Based on the analysis conducted, Support Vector Machine (SVM) emerged as the most effective model for predicting diabetes using basic health features. Among the models evaluated SVM, Decision Tree, and K-Nearest Neighbors (KNN). SVM demonstrated the highest recall and F1-score for the diabetic class. In medical applications, where identifying true positive cases (i.e., individuals with diabetes) is of paramount importance, high recall is critical to minimizing false negatives.
While KNN provided a more balanced performance across both classes, and Decision Tree offered simplicity and interpretability, the SVM model proved superior in capturing diabetic cases with greater reliability. Given that the objective of this project is to support early diagnosis and intervention, SVM's strong performance in identifying diabetic individuals makes it the recommended model for deployment or further development.
Future improvement can be expanding the dataset for greater generalizability.

## CONCLUSION
This project successfully demonstrates the feasibility of using basic, non-invasive health data such as age, gender, height, weight, blood sugar level and BMI to predict the likelihood of diabetes using machine learning techniques. By focusing on accessible features, the approach aims to support early screening and intervention, particularly in low-resource settings where lab tests may not be readily available.
Among the various models evaluated, Support Vector Machine (SVM) stood out as the most effective classifier, particularly in its ability to correctly identify individuals with diabetes. This high recall performance is crucial in healthcare applications where failing to detect a true positive case can lead to serious health consequences.
Key takeaways include:
-	Feature Simplicity: With only a few health indicators, the model achieves meaningful predictive power without complex data inputs.
-	Model Selection: SVM was chosen over Decision Tree and KNN due to its superior performance in recall and F1-score for the diabetic class.
-	Scalability: Although the dataset is relatively small, the model provides a solid foundation for scaling with larger and more diverse datasets in the future.
-	Hyperparameter Tuning: While tuning did not yield improvements over the baseline SVM, it highlighted the model's robustness and resistance to overfitting with limited data.
This project highlights the potential of machine learning to aid in early disease detection using minimal input data. It lays the groundwork for more advanced health risk assessment tools and encourages future work in expanding datasets, incorporating more features, and testing ensemble methods or deep learning models for further gains.
By providing a lightweight, interpretable, and high-performing model, this solution can serve as a valuable tool for preventive healthcare and digital health platforms aimed at managing chronic diseases like diabetes.

## IMPACT
This project demonstrates that even with a minimal and accessible set of features, it is possible to build an effective machine learning model for diabetes prediction. It showcases the value of data-driven decision-making in preventive healthcare and lays the groundwork for building lightweight screening tools.


















