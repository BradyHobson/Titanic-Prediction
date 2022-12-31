"""
Titanic Dataset Analysis and Beginner Machine Learning
By: Brady Hobson
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV

FILENAME = "titanic.csv"

def read_csv(filename):
    """
    Read in Titanic CSV into dataframe
    :param filename: str of filename
    :return: dataframe
    """
    # read and return csv
    df = pd.read_csv(filename)
    return df

def update_df(df):
    """
    Simple Data Manipulation
    :param df: dataframe
    :return: updated dataframe
    """
    # Turn sex data into numerical data
    enc = pd.get_dummies(df["Sex"])
    df = df.join(enc)

    # change datatypes to int64
    df['female'] = df['female'].astype("int64")
    df['male'] = df['male'].astype("int64")
    return df

def gen_surv(df):
    """
    Data Visualization of Graph of Survival Status Based on Gender
    :param df: Titanic Dataframe
    :return: Double Bar Graph
    """
    # Get death and survival counts for each gender
    graph_df = df[['Sex','Survived']]
    gender_counts = graph_df[['Sex','Survived']].value_counts().reset_index(name="Count")

    # Create new dataframe to use to graph with
    bar_df = pd.DataFrame({
        "Status": ["Survived", "Died"],
        "Men": [gender_counts["Count"][2], gender_counts["Count"][0]],
        "Women": [gender_counts["Count"][1], gender_counts["Count"][3]]
    })

    # Plot the dataframe into a double bar graph
    bar_df.plot(x="Status", y=["Men", "Women"], kind="bar")
    plt.title("Survival Status Based on Gender")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def exploratory_plots(df):
    """
    Creating Exploratory Plots to Explain the Best Categories
    :param df: Titanic Dataframe
    :return: Exploratory Graph
    """
    # new dataframe without names
    heatmap_df = df[["Survived", "Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare", "female", "male"]]

    # create heatmap of correlations
    sns.heatmap(heatmap_df.corr())
    plt.title("Correlation Heatmap of Titanic Passengers")
    plt.tight_layout()
    plt.show()

    # new dataframe with economic variables
    pair_df = df[["Survived", "Pclass", "Fare"]]

    # pair plot of exploratory data
    sns.pairplot(pair_df)
    plt.title("Economic Pair Plot")
    plt.show()


def knn(df, variable_lst = []):
    """

    :param df:
    :param variable_lst:
    :return:
    """
    if len(variable_lst) > 0:
        pass
    else:
        variable_lst = ["Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare", "female", "male"]

    X_train, X_test, y_train, y_test = train_test_split(df[variable_lst],
                                                        df["Survived"],
                                                        test_size=0.3,
                                                        random_state=7)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    f1_score(y_test, prediction)
    report = classification_report(y_test, prediction)
    return report, prediction

def find_best_k(X_train, X_test, y_train, y_test):
    # build the k-nn model, experiment with different values of k and plot the results
    accuracy = []
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        prediction = knn.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))

    return accuracy.index(max(accuracy))

if __name__ == "__main__":
    # Create Main Dataframe
    titanic_df = read_csv(FILENAME)
    titanic_df = update_df(titanic_df)

    # Create Data Visualization and Exploratory Plots
    gen_surv(titanic_df)
    exploratory_plots(titanic_df)

    # Base KNN Model
    report, prediction = knn(titanic_df)
    # print("KNN predictions: ", prediction)
    print("KNN Report:", report)

    #Prompt
    input("Hello:")







