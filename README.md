# Projects for GEN AI learning.


#### Pre-Requisites

1. Open a new folder for your projects.
2. Create the env - python -m venv genaienv
3. Create .env file.
4. Create requirements.txt for all of your installations.
5. Go to Langchain -> Login -> Create a new Project -> Create API keys
6. Put the LANGCHAIN_PROJECT and LANGCHAIN_API_KEY in your .env file.
7. Go to Groq.com -> login -> Create an API key -> Copy paste that in GROQ_API_KEY in .env file.


### 1. Simple LCEL App 
##### Build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting.

1. Create new file - simplellmLCEL.ipynb to execute your code one by one. Then finalize the whole code in serve.py file. 
2. Initializes the Language Model: It retrieves a Groq API key from the environment variables and sets up a ChatGroq instance using the llama-3.1-8b-instant model.
3. Defines the Prompt Templates: It creates a structured conversation template (ChatPromptTemplate) consisting of a system message instructing the model to translate text into a specified language, and a user message containing the text to be translated.
4. Sets up an Output Parser: It initializes a StrOutputParser to cleanly convert the model's raw chat response into a standard string.
5. Constructs the LCEL Chain: It links the prompt template, the language model, and the output parser together using the LangChain Expression Language (|) pipe syntax to form a single execution chain.
6. Configures the FastAPI Application: It creates a FastAPI backend server instance with custom metadata (title, version, and description).
7. Exposes the Chain as an API Route: It uses LangServe's add_routes function to automatically deploy the LangChain translation chain as a fully functional API endpoint at the /chain path.
8. Launches the Server: It triggers the Uvicorn ASGI server to run the application locally when the script is executed directly.



