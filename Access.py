import pandas as pd
from sqlalchemy import create_engine

# Database connection settings
db_config = {
    'host': '127.0.0.1',
    'user': 'root',         # Your MySQL username
    'password': '',         # Your MySQL password (empty by default in XAMPP)
    'database': 'database_1'  # Your database name
}

def get_dataframe():
    """
    Connect to the MySQL database, retrieve data, and return it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns None if there is an error.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

        # Define your SQL query
        query = "SELECT * FROM biplot;"  # Ensure 'biplot' is the correct table name

        # Load data into a Pandas DataFrame
        df = pd.read_sql(query, engine)

        # Adjust Pandas display options
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to multiple lines

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # If this script is run directly, display the DataFrame
    df = get_dataframe()
    if df is not None:
        print(df)
        print(df.info())
        print(df.describe(include='all'))
    else:
        print("Failed to retrieve data.")
