# Ball Classification case study.

## 1. Problem Statement
Ball Classification case study is to **classify a ball** as either a **tennis ball** or a **cricket ball** based on its weight and **surface type**.

This is simple Machine Learning case study to understand how classification works.

## 2. Dataset
We are using a **small dataset of balls** with two features:  

| Weight (grams) | Surface | Label        |
|----------------|---------|-------------|
| 35             | Rough   | Tennis Ball |
| 47             | Rough   | Tennis Ball |
| 90             | Smooth  | Cricket Ball |
| 48             | Rough   | Tennis Ball |
| 92             | Smooth  | Cricket Ball |
| 35             | Rough   | Tennis Ball |
| 90             | Smooth  | Cricket Ball |
| 35             | Rough   | Tennis Ball |
| 35             | Rough   | Tennis Ball |
| 35             | Rough   | Tennis Ball |

**Feature Encoding:**  
- Surface: `Rough = 1`, `Smooth = 0`  

**Label Encoding:**  
- Tennis Ball = 1  
- Cricket Ball = 2  

---

## 3. Approach
- We are using a **Decision Tree Classifier** from `sklearn` (Scikit-learn: an open-source Python library that provides various Machine Learning algorithms).  
- Features (weight and surface) are used to **train the model**.  
- The trained model **predicts the type of ball** based on user input.  

This simple case study helps understand:  
- How **features and labels** are used in ML.  
- How to **train a classifier** and make predictions.  

---

## 4. Model Training / Code
The training is done directly in the code using the inbuild functions:

```python
from sklearn import tree

# Features and Labels
Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]
Labels = [1,1,2,1,2,1,2,1,1,1]

# Train Decision Tree
obj = tree.DecisionTreeClassifier()
obj = obj.fit(Features, Labels) 

```

## 5. How to Run

- Make sure **Python** and the **scikit-learn** library are installed.  
  You can install scikit-learn using:
  ```bash
  pip install scikit-learn
- Run the program using:
    ```bash
    python ball_classification.py

## 6. Conclusion
This case study shows how a simple Machine Learning model using a **Decision Tree Classifier** **learns from data (training)** and **makes predictions (classification)**.  
It demonstrates the basic idea of how ML works using a small dataset and simple features.