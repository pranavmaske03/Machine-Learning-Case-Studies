import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def IrisClassifier(data_path):

    data = pd.read_csv(data_path,index_col = 0)
    print("Size of Actual dataset ",len(data))

    # features and labels
    features = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
    labels = data["variety"]

    # split data into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.5)

    print("Training size: ",len(x_train))
    print("Testing size: ",len(x_test))

    # Decide the algorithm and train the model
    obj = tree.DecisionTreeClassifier(random_state = 42)
    obj = obj.fit(x_train, y_train)


    # predictions   
    y_pred = obj.predict(x_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nSample predictions:")
    results = pd.DataFrame({"Actual": y_test.values[:5], "Predicted": y_pred[:5]})
    print(results)


def main():
    print("------------Iris Flower Classification Case Study--------------")
    IrisClassifier("iris.csv")
    

if __name__ == "__main__":
    main()