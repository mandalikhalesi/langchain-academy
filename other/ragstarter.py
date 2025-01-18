import os

#deprecated - from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
# deprecated - from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
# deprecated from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# Unused - from langchain.chains import LLMChain
# Unused - from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "voc-384409096126677353625166f241add0e168.98359862"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

# Initialize your LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature="0.0", max_tokens="2000")

# Load your documents
loader = CSVLoader(file_path = './tv-reviews.csv')
docs = loader.load()
#print(docs)

# Use a Text Splitter to split the documents into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_splitter = splitter.split_documents(docs)

# Initialize your embeddings model
embeddings = OpenAIEmbeddings()

# Populate your vector database with the chunks
db = Chroma.from_documents(doc_splitter, embeddings)

QUERY = """
    Based on the reviews in the context, tell me what people liked about the picture quality.
    Make sure you do not paraphrase the reviews, and only use the information provided in the reviews.
    """
# Find top 5 semantically similar documents to the query
results = db.similarity_search(QUERY, k=5)

# Query your LLM with the query and the top 5 documents
prompt = PromptTemplate(
    template = "{query}\nContext: {context}",
    input_variables = ["query", "context"]
)

chain = load_qa_chain(llm, prompt=prompt, chain_type="stuff")
print("Output of chain >> ", chain.run(input_documents=results, query=QUERY))

#output = llm.invoke(prompt)
#print("Output >>   ", output)

