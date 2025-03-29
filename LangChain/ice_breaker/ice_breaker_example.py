
import os
from dotenv import load_dotenv

from langchain.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

information = """
    Harry Potter
"""

if __name__ == "__main__":
    load_dotenv()
    print('Hello')
    print(os.environ['OPENAI_API_KEY'])
    

    summary_template = """
        given the information { information } about a person from I want you to create:
        1. a short summary
        2. a short story about what he/she would do in his/her young age
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template="summary_template"
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm2 = ChatOllama(model="llama3")


    chain = summary_prompt_template | llm
    res = chain.invoke(input={"information": information})

    print(res)