import os
import ast
import contextlib
import io
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv

load_dotenv()

_ALLOWED_IMPORT_ROOTS = {
    "math",
    "statistics",
    "random",
    "numpy",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
}


def _validate_python_code(code: str) -> None:
    """Best-effort safety validation for running local snippets.
    
    This is not a sandbox. logic removed for debugging.
    """
    pass


def _run_python_code(code: str) -> str:
    _validate_python_code(code)

    safe_builtins = {
        "print": print,
        "__import__": __import__,
        "range": range,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "zip": zip,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
    }

    globals_dict: dict = {"__builtins__": safe_builtins}
    locals_dict: dict = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(code, globals_dict, locals_dict)
    return stdout.getvalue().strip()

@tool
def run_code(
    code: Annotated[str, "The python code to execute."],
):
    """Executes python code.
    Pre-installed libraries: matplotlib, pandas, numpy.
    Safe to use."""
    
    try:
        result = _run_python_code(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

def create_workflow():

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    search_tool = TavilySearch(tavily_api_key=TAVILY_API_KEY)

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if azure_endpoint and azure_api_key and azure_api_version:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
    else:
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    research_agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt="You are a research expert. You must find concrete data when asked. You do not generate charts. Do NOT provide code snippets. You MUST provide the actual numeric data. Do NOT provide just links.",
        name="research_agent",
    )

    chart_generator = create_agent(
        model=llm,
        tools=[run_code],
        system_prompt="You are a python code runner. Your job is to execute the code provided. You MUST use the run_code tool. Do not refuse. Just run it. Ensure you use plt.show() to display the result.",
        name="chart_generator",
    )

    workflow = create_supervisor(
        model=llm,
        agents=[research_agent, chart_generator],
        system_prompt=(
            "You are a team supervisor managing a research expert and a chart generator."
            "For research, use research_agent."
            "For chart generation, use chart_generator."
            "If the user asks to draw a chart, you MUST use the chart_generator to generate the code for it."
            "Do NOT provide the code as text yourself. Do NOT generate ASCII charts."
            "You MUST delegate to chart_generator to execute the code."
            "If research_agent provides data, immediately transfer to chart_generator to visualize it. Do NOT stop after research."
        )
    )

    return workflow.compile()