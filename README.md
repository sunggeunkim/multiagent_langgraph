
## Configure LLM

This project uses `langchain-openai`.

### Default OpenAI

Set:

- `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL` (default: `gpt-4o`)

### Azure OpenAI

If all of the following are set, the app will use Azure OpenAI automatically:

- `AZURE_OPENAI_ENDPOINT` (e.g. `https://<resource-name>.openai.azure.com`)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (e.g. `2024-10-21`)
- `AZURE_OPENAI_DEPLOYMENT` (your Azure OpenAI deployment name for chat)

Notes:

- Azure uses a *deployment name* instead of an OpenAI `model=` name.
- Put these in a `.env` file if you want (this repo calls `load_dotenv()`).
