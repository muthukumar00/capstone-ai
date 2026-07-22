import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import tool
from langchain.agents import create_agent

from pydantic import BaseModel


# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")

model = init_chat_model(
    MODEL,
    model_provider="openrouter",
    temperature=0,
)

print("=" * 50)
print("      Meridian Bank Assistant")
print("=" * 50)


# =====================================================
# Part A - Prompt Template Chain
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Meridian Bank's assistant. "
            "Answer in one or two short sentences.",
        ),
        ("human", "{question}"),
    ]
)

chain = prompt | model | StrOutputParser()


questions = [
    "What documents are required for a home loan?",
    "How can I open a savings account?",
]

print("\n------ PART A ------")

for q in questions:
    print(f"\nQ: {q}")
    answer = chain.invoke({"question": q})
    print("A:", answer)


# =====================================================
# Part B - EMI Tool + Agent
# =====================================================

@tool
def calculate_emi(principal: float, annual_rate: float, years: int) -> dict:
    """
    Calculate the monthly EMI for a loan.
    """

    r = annual_rate / 100 / 12
    n = years * 12

    if r == 0:
        emi = principal / n
    else:
        emi = (
            principal
            * r
            * (1 + r) ** n
            / ((1 + r) ** n - 1)
        )

    return {"emi": round(emi, 2)}


agent = create_agent(
    model=model,
    tools=[calculate_emi],
    system_prompt=(
        "You are Meridian Bank's assistant. "
        "Use tools whenever EMI calculation is needed."
    ),
)


print("\n------ PART B ------")

question = (
    "What is the EMI on a loan of 5000000 "
    "at 8.4 percent for 25 years?"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }
)

print("\nQ:", question)
print("A:", result["messages"][-1].content)


# =====================================================
# Part C - Structured Output
# =====================================================

print("\n------ PART C ------")


class LoanRequest(BaseModel):
    intent: str
    product: str
    amount: str


structured_model = model.with_structured_output(LoanRequest)

loan_info = structured_model.invoke(
    "I'd like a home loan of about 30 lakh."
)

print("Structured Output:")
print(loan_info)