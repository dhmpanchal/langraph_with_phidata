from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

CATEGORIES = [
    "Patient basic info",
    "Vitals",
    "Diagnosis",
    "Medications",
    "Hospital admission",
    "Safeguarding",
    "Other"
]

class MedicalCategoryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="llama3.2",
            base_url="http://localhost:11434/v1",  # Ollama endpoint
            api_key="ollama",
            temperature=0
        )

        self.prompt = SystemMessage(
            content="""
            You are a medical document classifier.

            Your task is to read the provided clinical text chunk and classify it into ONE of the following categories:

            {categories}

            Rules:
            - Respond with ONLY the category name
            - No explanation
            - If nothing matches, return "Other"

            TEXT:
            {content}
            """
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=self.prompt,
        )

    def classify_chunk(self, content: str) -> str:
        try:
            content = f"""
            Content:
            {content}

            Categories:
            {"\n".join(CATEGORIES)}
            """
            input_message = HumanMessage(content=content)

            res = self.agent.invoke({
               "messages": [
                    input_message
                ]
            })

            category = res["messages"][-1].content.strip()
            print(f"Category: {category}")

            if category not in CATEGORIES:
                return "Other"

            return category
        except Exception as e:
            print(f"Error classifying chunk: {e}")
            return "Other"
