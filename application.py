"""
Titanic Dataset Analysis and Beginner Machine Learning
Using dataset from https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html
By: Brady Hobson
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

FILENAME = "titanic.csv"

def read_csv(filename):
    """
    read in Titanic CSV into dataframe
    :param filename: str of filename
    :return: dataframe
    """
    # read and return csv
    df = pd.read_csv(filename)
    return df

def update_df(df):
    """
    simple data manipulation
    :param df: Titanic dataframe
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
    data visualization of graph of survival status based on gender
    :param df: Titanic dataframe
    :return: double bar graph
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
    Creating exploratory plots to explain the best categories
    :param df: Titanic dataframe
    :return: exploratory graph
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
    pp = sns.pairplot(pair_df)
    pp.fig.suptitle("Economic Pair Plot")
    plt.tight_layout()
    plt.show()


def knn(df, variable_lst = []):
    """
    Create a K Nearest Neighbors model using train test split to predict survival on the Titanic
    :param df: Titanic dataframe
    :param variable_lst: optional list of variables for the model
    :return: classification report, prediction array
    """
    # check if user inputs its own variable list
    if len(variable_lst) > 0:
        pass
    else:
        variable_lst = ["Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "female", "male"]

    # Create train test split
    X_train, X_test, y_train, y_test = train_test_split(df[variable_lst],
                                                        df["Survived"],
                                                        test_size=0.3,
                                                        random_state=7)

    # find the best k for KNN model
    best_k = find_best_k(X_train, X_test, y_train, y_test)

    # Run KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    report = classification_report(y_test, prediction)
    return report, prediction, best_k

def find_best_k(X_train, X_test, y_train, y_test):
    """
    Find the best K for a knn model
    :param X_train: training data
    :param X_test: test data
    :param y_train: training data
    :param y_test: test data
    :return: best K
    """


    # figure out the best accuracy for different K values
    accuracy = []
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        prediction = knn.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))

    return (accuracy.index(max(accuracy)) + 1)

def predict_survival(person_dict, df, best_k):
    """
    Use KNN model to predict survival of the person
    :param person_dict: dictionary of inputted person
    :param df: Titanic dataframe
    :return: survival prediction
    """
    # create person dataframe
    person_df = make_dataframe(person_dict)

    # split data
    cols = [x for x in list(person_df.columns)]
    x_train = df.loc[:, cols].values
    y_train = df.loc[:, "Survived"].values

    x_test = person_df.loc[:, cols].values

    # Run KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)

    # report prediction
    if prediction[0] == 1:
        print("It is predicted that " + person_dict["name"] + " would survive on the Titanic")
    else:
        print("It is predicted that " + person_dict["name"] + " would not survive on the Titanic")

def make_dataframe(person_dict):
    """
    Make sure every value was put in correctly and create dataframe
    :param person_dict: dictionary of inputted person
    :return: dataframe with correct columns and values
    """
    # determine age variable
    age_drop = False
    try:
        age = int(person_dict["age"])
    except ValueError:
        age = 0
        age_drop = True

    # determine sex variable
    person_dict["sex"] = person_dict["sex"].lower()
    sex_drop = False
    if person_dict["sex"] == "male":
        male = 1
        female = 0
    elif person_dict["sex"] == "female":
        female = 0
        male = 1
    else:
        female = 0
        male = 0
        sex_drop = True

    # determine sibling/spouce value
    try:
        sib_spouce = int(person_dict["siblings"]) + int(person_dict["spouce"])
    except ValueError:
        try:
            sib_spouce = int(person_dict["siblings"])
        except ValueError:
            try:
                sib_spouce = int(person_dict["spouce"])
            except ValueError:
                sib_spouce = 0

    # determine parents/children value
    try:
        parent_child = int(person_dict["parents"]) + int(person_dict["children"])
    except ValueError:
        try:
            parent_child = int(person_dict["parents"])
        except ValueError:
            try:
                parent_child = int(person_dict["children"])
            except ValueError:
                parent_child = 0

    # determine Pclass variable
    pclass_drop = False
    try:
        pclass = int(person_dict["pclass"])
        if (pclass < 1) | (pclass > 3):
            pclass = 0
            pclass_drop = True
    except ValueError:
        pclass = 0
        pclass_drop = True

    # create dataframe
    person_df = pd.DataFrame({
        "Age": age,
        "male": male,
        "female": female,
        "Siblings/Spouses Aboard": sib_spouce,
        "Parents/Children Aboard": parent_child,
        "Pclass": pclass
    }, index=[0])

    # drop columns if needed
    if age_drop == True:
        person_df = person_df.drop("Age", axis=1)

    if sex_drop == True:
        person_df = person_df.drop("male", axis=1)
        person_df = person_df.drop("female", axis=1)

    if pclass_drop == True:
        person_df = person_df.drop("Pclass", axis=1)

    return person_df

if __name__ == "__main__":
    # Create Main Dataframe
    titanic_df = read_csv(FILENAME)
    titanic_df = update_df(titanic_df)

    # Create Data Visualization and Exploratory Plots
    gen_surv(titanic_df)
    exploratory_plots(titanic_df)

    # Base KNN Model
    report, prediction, best_k = knn(titanic_df)
    print("Using a Train Test Split on our data I used a K Nearest Neighbors model to predict survival on the Titanic.")
    print("The KNN Report:", report)

    # prompt the user
    print("To predict survival on the titanic. Please enter the following information.")
    name = input("Name: ")
    age = input("Age: ").strip()
    sex = input("Sex (Male or Female): ").strip()
    siblings = input("Number of Siblings: ").strip()
    spouce = input("Number of Spouces: ").strip()
    parents = input("Number of Parents: ").strip()
    children = input("Number of Children: ").strip()
    pclass = input("Passenger Class (1 = Upper Class, 2 = Middle Class, 3 = Lower Class): ").strip()

    # create dictionary of the values
    person_dict = {"name": name, "age": age, "sex": sex, "siblings": siblings,
                   "spouce": spouce, "parents": parents, "children": children, "pclass": pclass}

    # predict survival of the added person
    predict_survival(person_dict, titanic_df, best_k)