import os
# from langchain.llms import OpenAI # this code has been deprecated since recording.
from random import sample 
from typing import List

# from langchain.chat_models import ChatOpenAI # this is the replacement 
from langchain_openai import ChatOpenAI
#deprecated - from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt

os.environ["OPENAI_API_KEY"] = "voc-384409096126677353625166f241add0e168.98359862"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

# Load reviews from tv-reviews.csv

loader = CSVLoader(file_path = './tv-reviews.csv')
data = loader.load()

print(data)

# Initialize OpenAI object with your API key
MODEL = "gpt-4o-mini"
#MAX_TOKENS=2000
TEMPERATURE="0.0"
llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

class ReviewSentiment(BaseModel):
    "Estimates review sentiment of an input using a base model"
    positives: List[NonNegativeInt] = Field(description="index of a positive TV review, starting from 0")
    negatives: List[NonNegativeInt] = Field(description="index of a negative TV review, starting from 0")
        
parser = PydanticOutputParser(pydantic_object=ReviewSentiment)

# Setup a template with partial and input variables
print(parser.get_format_instructions())

prompt = PromptTemplate(
    template = "{question}\n{format_instructions}\nContext: {context}",
    input_variables = ["question", "context"],
    partial_variables = {"format_instructions": parser.get_format_instructions},
)

question = """
    Classify TV reviews provided in this context into positive and negative.
    Only use the reviews provided in this context.
    If there are no positive or negative reviews, revert an empty JSON array.
"""

# Pick 3 random reviews and save them into reviews_to_classify variable
reviews_to_classify = sample(data, 3)
context = '\n'.join(review.page_content for review in reviews_to_classify)
query = prompt.format(context=context, question=question)
print(query)

# Query LLM, then parse output into the result variable
output = llm.invoke(query)
print(output)
result = parser.parse(output)
# Seems that the input data is not a valid string...
# pydantic_core._pydantic_core.ValidationError: 1 validation error for Generation
# text
#  Input should be a valid string [type=string_type, input_value=AIMessage(content='```jso...8, 'total_tokens': 528}), input_type=AIMessage]
#    For further information visit https://errors.pydantic.dev/2.9/v/string_type

print(result)
print("Positives:\n" + "\n".join([reviews_to_classify[i].page_content for i in result.positives]))
print("Negatives:\n" + "\n".join([reviews_to_classify[i].page_content for i in result.negatives]))

