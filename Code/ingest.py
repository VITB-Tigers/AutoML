from db_utils import write_to_mongo

# Ingest data from a file and store it in MongoDB.
def ingest_data(data, collection_name):
    """
    :param file_path: Path to the dataset file.
    :param collection_name: Name of the MongoDB collection where data will be stored.
    """

    # Write to MongoDB
    write_to_mongo(data, collection_name)

if __name__ == "__main__":
    pass