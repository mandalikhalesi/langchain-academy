from langchain.indexes import VectorstoreIndexCreator
#deprecated - from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='./tv-reviews.csv')

index = VectorstoreIndexCreator().from_loaders([loader])

query = "Based on the reviews in the context, tell me what people liked about the picture quality"
index.query(query)

# Getting an error - 
# UserWarning: Using InMemoryVectorStore as the default vectorstore.
# This memory store won't persist data.
# You should explicitlyspecify a vectorstore when using VectorstoreIndexCreator
