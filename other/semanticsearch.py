import os

#deprecated - from langchain.document_loaders.csv_loader import CSVLoader
#from langchain_community.document_loaders import CSVLoader
#deprecated - from langchain.llms import openai
from langchain_community.llms import openai
#from langchain.text_splitter import CharacterTextSplitter
# deprecated - from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#deprecated - from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "voc-384409096126677353625166f241add0e168.98359862"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

model_name = "gpt-4o-mini"
temperature = "1.2"
llm = ConversationChain{
    llm=llm,
    memory = ConversationBufferMemory(),
    verbose=true
}