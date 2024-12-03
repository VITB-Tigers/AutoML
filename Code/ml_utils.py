import pandas as pd
from sklearn.model_selection import train_test_split
from db_utils import read_from_mongo, write_to_mongo
from utils import load_config
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression

config = load_config()
X_train_collection_name = config["mongo"]["X_train_collection_name"]
y_train_collection_name = config["mongo"]["y_train_collection_name"]
X_test_collection_name = config["mongo"]["X_test_collection_name"]
y_test_collection_name = config["mongo"]["y_test_collection_name"]

def split_data_and_save_to_mongo(collection_name, target, train_size):
    """
    Split data retrieved from MongoDB into training and testing sets and save them back to MongoDB.

    Parameters:
    collection_name (str): Name of the MongoDB collection to read data from and save split data.
    target (str): The name of the target variable column.
    train_size (float): The proportion of the dataset to include in the train split.
    """
    # Retrieve the data from MongoDB
    data = read_from_mongo(collection_name)
    data = pd.DataFrame(data)
    print(data.head())

    # Separate the features (X) and the target variable (y)
    X = data.drop(columns=[target])
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Convert DataFrame and Series to dictionaries/lists
    X_train = X_train.to_dict("records")
    X_test = X_test.to_dict("records")
    y_train = [{'target': val} for val in y_train]
    y_test = [{'target': val} for val in y_test]

    # Save the data into MongoDB
    write_to_mongo(X_train, X_train_collection_name)
    write_to_mongo(X_test, X_test_collection_name)
    write_to_mongo(y_train, y_train_collection_name)
    write_to_mongo(y_test, y_test_collection_name)

    print("Data successfully split and saved to MongoDB.")

# Function to get the accuracy of the model
def get_model_accuracy(model, X_test_collection_name, y_test_collection_name):
    """
    Get the accuracy of a model using the test data.
    
    Parameters:
    model: The trained model to evaluate.
    X_test_collection_name (str): Name of the MongoDB collection containing the test features.
    y_test_collection_name (str): Name of the MongoDB collection containing the test target variable.
    """
    # Load the test data from MongoDB
    X_test = pd.DataFrame(read_from_mongo(X_test_collection_name))
    y_test = pd.DataFrame(read_from_mongo(y_test_collection_name))
    
    # Drop the '_id' column from the data
    X_test.drop('_id', axis=1, inplace=True)
    y_test.drop('_id', axis=1, inplace=True)
    
    # Get the accuracy of the model
    accuracy = model.score(X_test, y_test)
    
    return accuracy

# Train a Linear Regression model using the training data while taking the parameters: Fit_intercept, n_jobs
def train_linear_regression(X_train_collection_name, y_train_collection_name, fit_intercept, n_jobs):
    """
    Train a Linear Regression model using the training data.
    
    Parameters:
    X_train_collection_name (str): Name of the MongoDB collection containing the training features.
    y_train_collection_name (str): Name of the MongoDB collection containing the training target variable.
    fit_intercept (bool): Whether to calculate the intercept for this model.
    n_jobs (int): The number of jobs to use for the computation.
    """
    # Load the training data from MongoDB
    X_train = pd.DataFrame(read_from_mongo(X_train_collection_name))
    y_train = pd.DataFrame(read_from_mongo(y_train_collection_name))
    
    # Drop the '_id' column from the data
    X_train.drop('_id', axis=1, inplace=True)
    y_train.drop('_id', axis=1, inplace=True)
    
    # Instantiate the Linear Regression model
    model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
    
    # Train the model
    model.fit(X_train, y_train)
    
    test_accuracy = get_model_accuracy(model, X_test_collection_name, y_test_collection_name)
    train_accuracy = get_model_accuracy(model, X_train_collection_name, y_train_collection_name)
    
    return model, test_accuracy, train_accuracy

# Train a Decision Tree Classifier model using the training data while taking the parameters: Min_samples_split, Max_depth, and Criterion    
def train_decision_tree_classifier(X_train_collection_name, y_train_collection_name, criterion, max_depth, min_samples_split):
    """
    Train a Decision Tree Classifier model using the training data.
    
    Parameters:
    X_train_collection_name (str): Name of the MongoDB collection containing the training features.
    y_train_collection_name (str): Name of the MongoDB collection containing the training target variable.
    criterion (str): The function to measure the quality of a split. 
    max_depth (int): The maximum depth of the tree.
    min_samples_split (int): The minimum number of samples required to split an internal node.
    """
    # Load the training data from MongoDB
    X_train = pd.DataFrame(read_from_mongo(X_train_collection_name))
    y_train = pd.DataFrame(read_from_mongo(y_train_collection_name))
    
    # Drop the '_id' column from the data
    X_train.drop('_id', axis=1, inplace=True)
    y_train.drop('_id', axis=1, inplace=True)
    
    # Instantiate the Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    test_accuracy = get_model_accuracy(clf, X_test_collection_name, y_test_collection_name)
    train_accuracy = get_model_accuracy(clf, X_train_collection_name, y_train_collection_name)
    
    return clf, test_accuracy, train_accuracy

# Train an AdaBoost Classifier model using the training data while taking the parameters: N_estimators, Learning_rate
def train_adaboost_classifier(X_train_collection_name, y_train_collection_name, n_estimators, learning_rate):
    """
    Train an AdaBoost Classifier model using the training data.
    
    Parameters:
    X_train_collection_name (str): Name of the MongoDB collection containing the training features.
    y_train_collection_name (str): Name of the MongoDB collection containing the training target variable.
    n_estimators (int): The maximum number of estimators at which boosting is terminated.
    learning_rate (float): The learning rate shrinks the contribution of each classifier.
    """
    # Load the training data from MongoDB
    X_train = pd.DataFrame(read_from_mongo(X_train_collection_name))
    y_train = pd.DataFrame(read_from_mongo(y_train_collection_name))
    
    # Drop the '_id' column from the data
    X_train.drop('_id', axis=1, inplace=True)
    y_train.drop('_id', axis=1, inplace=True)
    
    # Instantiate the AdaBoost Classifier
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    test_accuracy = get_model_accuracy(clf, X_test_collection_name, y_test_collection_name)
    train_accuracy = get_model_accuracy(clf, X_train_collection_name, y_train_collection_name)
    
    return clf, test_accuracy, train_accuracy

# Save model to the specified path using joblib
def save_model(model, model_path):
    """
    Save a model to the specified path using joblib.
    
    Parameters:
    model: The model to save.
    model_path (str): The path where the model will be saved.
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    model = train_adaboost_classifier(X_train_collection_name, y_train_collection_name, 50, 1.0)
    accuracy = get_model_accuracy(model, X_test_collection_name, y_test_collection_name)
    print(f"Model accuracy: {accuracy}")