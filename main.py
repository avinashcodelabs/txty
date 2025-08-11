from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import warnings


load_dotenv()

warnings.filterwarnings("ignore")

chat = ChatOpenAI(
    model="deepseek-coder:6.7b",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="doesn't matter",
    verbose=True
)


memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory(file_path="./messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")
    if content == "bye":
        break

    result = chain({
        "content": content,
    })
    print(result["text"])
