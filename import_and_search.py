from repo_embedder import search_functions
import pandas as pd

# Load the DataFrame from a CSV file
df_loaded = pd.read_csv('embeddings.csv')

# Convert list-like strings back to numpy arrays
df_loaded['code_embedding'] = df_loaded['code_embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Now you can search for functions
search_functions(df_loaded, "music parser")