import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.embeddings import init_embeddings
from pydantic import SecretStr

# Load environment variables
load_dotenv()

MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")

# Initialize the chat model
model = init_chat_model(
    MODEL,
    model_provider="openrouter",
    temperature=0,
)

# ============================================================
# Task 1 - Call the model with LangChain messages
# ============================================================

print("========== Task 1 ==========")

response = model.invoke([
    SystemMessage(
        content="You are a friendly banking assistant. Answer in one short sentence."
    ),
    HumanMessage(
        content="What is a floating interest rate?"
    )
])

print("Bot:", response.content)

# ============================================================
# Task 2 - Prompt Template + Chain
# ============================================================

print("\n========== Task 2 ==========")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Meridian Bank's assistant. Answer in one sentence."),
    ("user", "Explain '{term}' to a first-time customer."),
])

chain = prompt | model | StrOutputParser()

answer = chain.invoke({"term": "EMI"})
print("EMI ->", answer)

answer = chain.invoke({"term": "CIBIL score"})
print("CIBIL Score ->", answer)

# ============================================================
# Task 3 - Agent with Tool
# ============================================================

print("\n========== Task 3 ==========")

@tool
def calculate_emi(principal: float, annual_rate: float, years: int) -> dict:
    """Calculate the monthly EMI for a loan."""
    r = annual_rate / 100 / 12
    n = years * 12

    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    return {"emi": round(emi, 2)}

agent = create_agent(
    model=model,
    tools=[calculate_emi],
    system_prompt="You are Meridian Bank's assistant. Use tools for EMI maths.",
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What's the EMI on a 50 lakh loan at 8.4% over 25 years?"
        }
    ]
})

print(result["messages"][-1].content)

# ============================================================
# Task 4 - Bonus (Embeddings)
# ============================================================

print("\n========== Task 4 ==========")

embeddings = init_embeddings(
    "openai:text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    check_embedding_ctx_length=False,
)

vector = embeddings.embed_query("home loan eligibility")

print("Vector length:", len(vector))

# ============================================================
# Notes
# ============================================================

# Task 3:
# The agent decides whether to call the calculate_emi tool based on the user's request.

# Task 4:
# An embedding is a numerical representation of text that captures its meaning,
# allowing similar pieces of text to be compared.