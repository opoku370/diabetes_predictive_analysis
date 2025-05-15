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
-<a href= “             back”>Dataset </a>

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









