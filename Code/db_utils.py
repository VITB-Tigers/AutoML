# Utilities file for functions related to MongoDB

from utils import load_config
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder

# CRUD operations for MongoDB
def connect_mongo():
    """
    Connect to MongoDB
    """
    config = load_config()
    host = config["mongo"]["host"]
    port = config["mongo"]["port"]
    db = config["mongo"]["db"]
    username = config["mongo"]["username"]
    password = config["mongo"]["password"]
    
    connection_string = f'mongodb://{username}:{password}@{host}:{port}/'
    client = MongoClient(connection_string)
    
    return client[db]

def write_to_mongo(data, collection):
    """
    Write data to MongoDB 
    """
    db = connect_mongo()
    
    # Check if collection already exists
    if collection in db.list_collection_names():
        db[collection].drop()  # Drop the existing collection
    
    db[collection].insert_many(data)  # Insert the new data
    
    return "Data ingested successfully!"

def read_from_mongo(collection):
    """
    Read data from MongoDB
    """
    db = connect_mongo()
    data = list(db[collection].find())
    return data

def drop_db():
    """
    Drop the MongoDB database
    """
    db = connect_mongo()
    try:
        db.client.drop_database(db.name)
        return True
    except Exception as e:
        print(f"Error dropping database: {e}")
        return False


# MISC utilities
def get_collection_shape(collection):
    """
    Get the number of rows and columns in a MongoDB collection.
    
    :param collection: Name of the MongoDB collection.
    :return: Tuple (num_rows, num_columns) where num_rows is the number of documents,
             and num_columns is the number of unique fields across all documents.
    """
    db = connect_mongo()
    
    # Count the number of documents (rows)
    num_rows = db[collection].count_documents({})
    
    # Get all unique fields (columns)
    sample_doc = db[collection].find_one()
    if sample_doc:
        all_keys = set(sample_doc.keys())
        for doc in db[collection].find():
            all_keys.update(doc.keys())
        all_keys.discard("_id")  # Remove the "_id" column
        num_columns = len(all_keys)
    else:
        num_columns = 0
    
    return num_rows, num_columns

def get_column_names(collection):
    """
    Get the names of all columns in a MongoDB collection.
    
    :param collection: Name of the MongoDB collection.
    :return: List of column names.
    """
    db = connect_mongo()
    
    sample_doc = db[collection].find_one()
    if sample_doc:
        return [key for key in sample_doc.keys() if key != "_id"]
    else:
        return []

def remove_columns(collection, columns):
    """
    Remove columns from a MongoDB collection.
    
    :param collection: Name of the MongoDB collection.
    :param columns: List of columns to remove.
    """
    db = connect_mongo()
    
    for column in columns:
        db[collection].update_many({}, {"$unset": {column: ""}})

def encode_columns(collection, columns):
    """
    Encode columns in a MongoDB collection to numbers.
    
    :param collection: Name of the MongoDB collection.
    :param columns: List of columns to encode.
    """
    db = connect_mongo()
    
    for column in columns:
        label_encoder = LabelEncoder()
        all_values = [doc[column] for doc in db[collection].find()]
        label_encoder.fit(all_values)
        
        for doc in db[collection].find():
            encoded_value = label_encoder.transform([doc[column]])[0]
            db[collection].update_one({"_id": doc["_id"]}, {"$set": {column: int(encoded_value)}})

if __name__ == "__main__":
    data = [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]
    collection = "test"
    print(write_to_mongo(data, collection))
    print(read_from_mongo(collection))