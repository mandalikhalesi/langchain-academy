import os

#deprecated - from langchain.document_loaders.csv_loader import CSVLoader
#from langchain_community.document_loaders import CSVLoader
#from langchain.llms import openai
#from langchain.text_splitter import CharacterTextSplitter
# deprecated - from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#deprecated - from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain.docstore.document import Document

os.environ["OPENAI_API_KEY"] = "voc-384409096126677353625166f241add0e168.98359862"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

happy_doc = Document(page_content="""
                     Happiness is a pleasant and positive reaction, ranging from contentment to intense joy.
                     """)
football_doc = Document(page_content="""
                        Football is a game played with a spherical ball in accordannce with a set of rules known as the Laws of the Game.
                        """)
docs = [happy_doc, football_doc]

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embeddings)

query = """
        describes a feeling
        """

results = db.similarity_search(query, k=1)

print(results)
