import streamlit as st
import pandas as pd

from utils import read_file, load_config
from ingest import ingest_data
from ml_utils import split_data_and_save_to_mongo, train_linear_regression, train_decision_tree_classifier, train_adaboost_classifier, save_model
from db_utils import (get_collection_shape,
                      read_from_mongo,
                      drop_db, 
                      get_column_names,
                      remove_columns,
                      encode_columns)

config = load_config()
ingest_collection_name = config["mongo"]["ingest_collection_name"]
X_train_collection_name = config["mongo"]["X_train_collection_name"]
y_train_collection_name = config["mongo"]["y_train_collection_name"]
X_test_collection_name = config["mongo"]["X_test_collection_name"]
y_test_collection_name = config["mongo"]["y_test_collection_name"]

# Setting the page configuration for the web app
st.set_page_config(page_title="AutoML", page_icon=":bar_chart:", layout="centered")

# Adding a heading to the web app
st.markdown("<h1 style='text-align: center; color: white;'>AutoML </h1>", unsafe_allow_html=True)
st.divider()

# Creating tabs for the web app
tab1, tab2, tab3, tab4 = st.tabs(["Data Ingestion", "Data Transformation", "Auto Train ML Models", "Freeze the Learnings"])

# Tab for Data Ingestion
with tab1:
    st.subheader("Data Ingestion")
    
    with st.container(border=True):
        
        dataset_choice = st.radio("Choose an option to upload the data", ["Enter the path", "Upload the file"], horizontal=True)
        dataset_bool = True if dataset_choice == "Enter the path" else False
        
        file_location = st.text_input("Path of the file",
                                     placeholder="Dataset Path", 
                                     help="Enter the complete path to the source data",
                                     disabled=not dataset_bool)
        file_name = st.text_input("Name of the file",
                                     placeholder="Dataset Name",
                                     help="Enter the complete name with extension, i.e., .csv or .xlsx", 
                                     disabled=not dataset_bool)

        st.markdown("<h4 style='text-align: center;'>OR</h4>", unsafe_allow_html=True)

        dataset_upload = st.file_uploader("Upload the file",
                                     type=["csv", "xlsx"],
                                     help="Upload the file to ingest the data",
                                     disabled=dataset_bool)
            
        if st.button("Ingest", use_container_width=True):
            if dataset_bool:
                # If there is a / at the end of the path, remove it
                if file_location[-1] == "/":
                    file_location = file_location[:-1]
                file_path =f"{file_location}/{file_name}"
                data = read_file(file_path).to_dict("records")
            else :
                data = read_file(dataset_upload).to_dict("records")
            
            if data is not None:
                print(ingest_data(data, ingest_collection_name))
                if ingest_data is not None:
                    st.success("Data ingested successfully!", icon="✅")
                else:
                    st.error("Error ingesting data!", icon="❌")
            else:
                st.error("Error uploading data!", icon="❌")
    
    with st.form(key="Config Dimensions"):
        st.subheader("Data dimensions", help="Displays thee total number of columns in the dataset")

        rows, columns = get_collection_shape(ingest_collection_name)
        containter = st.container()

        if st.form_submit_button("Run", use_container_width=True):
            containter.write(f"Number of rows: {rows}")
            containter.write(f"Number of columns: {columns}")
            
            # Display the first few rows of the dataset
            data = read_from_mongo(ingest_collection_name)
            data = pd.DataFrame(data)
            if data.empty:
                st.error("No data found in the database!", icon="❌")
            else:
                data.drop("_id", axis=1, inplace=True) # Ignore the _id column as its just for MongoDB
                containter.dataframe(data.head(), hide_index=True)
            
    with st.container(border=True):
        st.subheader("Reset Database")
        if st.button("Drop Database", help="This will delete the database and all the data stored in it.", use_container_width=True):
            if drop_db():
                st.success("Database dropped successfully!", icon="✅")
            else:
                st.error("Error dropping database!", icon="❌")

# Tab for Data Transformation
with tab2:
    st.subheader("Data Transformation")

    current_rows = list(get_column_names(ingest_collection_name))

    with st.form(key="Config Remove"):
        removal_selection = st.multiselect("Remove features",
                                           options=current_rows,
                                           help="Enter the names of features to remove")
        
        if st.form_submit_button(label="Remove Features",use_container_width=True):
            remove_columns(ingest_collection_name, removal_selection)
            
            st.success("Feature(s) removed successfully!", icon="✅")
    
    with st.form(key="Config Convert"):
        st.multiselect("Convert to numbers",
                      options=current_rows, 
                      help="Enter the names of features to convert to numbers")
        
        if st.form_submit_button("Convert Feature(s)",use_container_width=True):
            encode_columns(ingest_collection_name, current_rows)
            st.success("Feature(s) converted successfully!", icon="✅")
            
    with st.form(key="Set Target Variable"):
        target_variable = st.selectbox("Target Variable",
                                       options=current_rows,
                                       placeholder="Target Variable",
                                       help="Enter the name of the target variable")
        
        if st.form_submit_button("Set Target Variable",use_container_width=True):
            st.success("Target variable set successfully!", icon="✅")    
    
    
    with st.form(key="Config Dim"):
        st.subheader("Data dimensions", help="Displays thee total number of columns in the dataset")

        rows, columns = get_collection_shape(ingest_collection_name)
        containter = st.container()

        if st.form_submit_button("Run", use_container_width=True):
            containter.write(f"Number of rows: {rows}")
            containter.write(f"Number of columns: {columns}")
            
            # Display the first few rows of the dataset
            data = read_from_mongo(ingest_collection_name)
            data = pd.DataFrame(data)
            data.drop("_id", axis=1, inplace=True) # Ignore the _id column as its just for MongoDB
            containter.dataframe(data.head(), hide_index=True)


linear_model = None
decision_tree_model = None
ada_boost_model = None

# Tab for Auto Training ML Models
with tab3:

    #TODO Might wanna shift train test split to data transformation tab
    with st.container(border=True):
        st.markdown("### Train Test Split Data")
        
        # Creating two columns for the form for visual appeal
        col1, col2 = st.columns(2)
        with col1:
            train_size = st.number_input("Training Data Split",
                          placeholder="% of training data",
                          help="Enter the percentage of data to be used for training",
                          value=70)
        
        test_size = 100 - train_size
        # This is just to visually show the test size
        with col2:
            test_size = st.number_input("Testing Data Split",
                          placeholder="% of testing data",
                          help="See the percentage of data to be used for testing",
                          value=test_size, disabled=True)
            
        if st.button("Split Data", use_container_width=True):
            data = read_from_mongo(ingest_collection_name)
            data = pd.DataFrame(data)
            train_size = train_size / 100
            split_data_and_save_to_mongo(ingest_collection_name, target_variable, train_size)
            st.success("Data split successfully!", icon="✅")
    
    st.divider()
    # Form for Linear Regression
    with st.form(key="Linear Regression"):
        st.markdown("### Linear Regresssion")

        # Expander for hyperparameters
        with st.expander("See more about hyperparameters", expanded=False):
            st.write("Lorem ipsum dolor sit amet")

        # Columns for form inputs
        col1, col2, = st.columns(2)

        with col1:
            fit_intercept = st.selectbox("Fit Intercept", options=["True", "False"], index=0, help="Whether to calculate the intercept for the model")
        with col2:
            n_jobs = st.number_input("N Jobs", min_value=-1, max_value=10, value=-1, help="Number of CPU cores to use for computations")
        
        # Convert fit_intercept to boolean
        fit_intercept = True if fit_intercept == "True" else False
        
        if st.form_submit_button("Run", use_container_width=True):
            linear_model, linear_test_accuracy, linear_train_accuracy = train_linear_regression(X_train_collection_name, y_train_collection_name, fit_intercept, n_jobs)
            st.text(f"Test Accuracy: {linear_test_accuracy:.2f}%")
            st.text(f"Train Accuracy: {linear_train_accuracy:.2f}%")
            st.success("Model trained on Linear Regression successfully!", icon="✅")

    # Form for Decision Tree
    with st.form(key="Decision Tree"):
        st.markdown("### Decision Tree")

        # Expander for hyperparameters
        with st.expander("See more about hyperparameters", expanded=False):
            st.write("Lorem ipsum dolor sit amet")

        # Columns for form inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            max_depth = st.number_input("Max Depth", min_value=1, value=10, help="Maximum depth of the tree")
        with col2:
            min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2, help="Minimum number of samples required to split an internal node")
        with col3:
            criterion = st.selectbox("Criterion", options=["gini", "entropy"], index = 0, help="Measure used to evaluate the quality of a split")

        if st.form_submit_button("Run", use_container_width=True):
            decision_tree_model, decision_tree_test_accuracy, decision_tree_train_accuracy = train_decision_tree_classifier(X_train_collection_name, y_train_collection_name, criterion, max_depth, min_samples_split)
            st.text(f"Test Accuracy: {decision_tree_test_accuracy:.2f}%")
            st.text(f"Train Accuracy: {decision_tree_train_accuracy:.2f}%")
            st.success("Model trained on Decision Tree successfully!", icon="✅")
            
    # Form for AdaBoost
    with st.form(key="AdaBoost"):
        st.markdown("### AdaBoost")

        # Expander for hyperparameters
        with st.expander("See more about hyperparameters", expanded=False):
            st.write("Lorem ipsum dolor sit amet")
        
        # Columns for form inputs
        col1, col2 = st.columns(2)

        with col1:
            n_estimators = st.number_input("N Estimators", min_value=1, value=50, help="Number of boosting stages to perform")
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.0, value=1.0, help="Shrinks the contribution of each classifier")

        if st.form_submit_button("Run", use_container_width=True):
            ada_boost_model, ada_boost_test_accuracy, ada_boost_train_accuracy = train_adaboost_classifier(X_train_collection_name, y_train_collection_name, n_estimators, learning_rate)
            st.text(f"Test Accuracy: {ada_boost_test_accuracy:.2f}%")
            st.text(f"Train Accuracy: {ada_boost_train_accuracy:.2f}%")
            st.success("Model trained on AdaBoost successfully!", icon="✅")

# Tab for Freezing the Learnings
with tab4:   
    st.subheader("Freeze the Learnings")
    
    with st.form("Linear Regression Freeze"):
        st.markdown("### Linear Regression")
        linear_model_path = st.text_input("Enter model path",
                                          help="Enter the path to save the model",
                                          value="Models/linear_model.pkl")
        
        if st.form_submit_button("Save Linear Regression Model", use_container_width=True):
            save_model(linear_model, linear_model_path)
            st.success("Learnings frozen successfully!", icon="✅")
            
    with st.form("Decision Tree Freeze"):
        st.markdown("### Decision Tree")
        decision_tree_model_path = st.text_input("Enter model path",
                                                 help="Enter the path to save the model",
                                                 value="Models/decision_tree_model.pkl")
        
        if st.form_submit_button("Save Decision Tree Model", use_container_width=True):
            save_model(decision_tree_model, decision_tree_model_path)
            st.success("Learnings frozen successfully!", icon="✅")
            
    with st.form("ADA Boost Freeze"):
        st.markdown("### ADA Boost")
        ada_boost_model_path = st.text_input("Enter model path",
                                            help="Enter the path to save the model",
                                            value="Models/ada_boost_model.pkl")
        
        if st.form_submit_button("Save ADA Boost Model", use_container_width=True):
            save_model(ada_boost_model, ada_boost_model_path)
            st.success("Learnings frozen successfully!", icon="✅")