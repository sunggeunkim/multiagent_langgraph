import os
import ast
import contextlib
import io
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
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

    This is not a sandbox. It blocks obvious risky operations and restricts imports
    to a small allowlist to avoid accidentally executing harmful code.
    """

    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            # For `import x as y`, each alias has a name
            names = [alias.name for alias in getattr(node, "names", [])]
            candidates = ([module] if module else []) + names
            for mod in filter(None, candidates):
                root = mod.split(".", 1)[0]
                if mod not in _ALLOWED_IMPORT_ROOTS and root not in _ALLOWED_IMPORT_ROOTS:
                    raise ValueError(
                        f"Import '{mod}' is not allowed. Allowed roots: {sorted(_ALLOWED_IMPORT_ROOTS)}"
                    )

        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile", "open", "input", "__import__"}:
            raise ValueError(f"Use of '{node.id}' is not allowed")

        # Block dunder attribute access patterns like obj.__class__
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")


def _run_python_code(code: str) -> str:
    _validate_python_code(code)

    safe_builtins = {
        "print": print,
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
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    try:
        result = _run_python_code(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

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
    system_prompt="You can only do research. You do not generate charts",
    name="research_agent",
)