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



### 2. Chatbot App (1-chatbot.ipynb) 
##### Design and implement an LLM-powered chatbot. This chatbot will be able to have a conversation and remember previous interactions. Note that this chatbot that we build will only use the language model to have a conversation.

1. Initializing Environment and LLM
This section sets up the required API keys and initializes the Large Language Model (LLM) utilizing Groq's hardware acceleration via LangChain.
(Code Explanation in the Code file)
2. Implementing Automated Message History (Stateful)
To avoid manually formatting and passing arrays of old messages, this section introduces an automated abstraction utility layer.
3. Prompt Templates and Complex Input Chains
This section transitions from sending raw lists of messages to structured components using LCEL.
4. Token Trimming & Conversation Optimization
As conversations extend, context windows eventually max out. Trimming messages forces constraints over your input lengths to manage computational budget.


### 2. Vector Store and Retriever (vectorretriever.ipynb) 
##### LangChain's vector store and retriever abstractions - These abstractions are designed to support retrieval of data-- from (vector) databases and other sources-- for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference, as in the case of retrieval-augmented generation.
(Detailed Code Summary in code file)
1. Document - LangChain implements a Document abstraction, which is intended to represent a unit of text and associated metadata. It has two attributes:
- page_content: a string representing the content;
- metadata: a dict containing arbitrary metadata.
The metadata attribute can capture information about the source of the document, its relationship to other documents, and other information. Note that an individual Document object often represents a chunk of a larger document.
2. Querying the Vector Database
Demonstrates different methods to query the collection based on contextual and structural meanings instead of exact keyword matching.
3. Converting Vector Stores to Retrievers
While VectorStore objects store and query data, they lack the standardized Runnable methods needed to link directly into LangChain Expression Language (LCEL) chains. A Retriever acts as a unified data fetching interface.
4. The End-to-End Retrieval-Augmented Generation (RAG) PipelineThis ties everything together into a functional RAG pipeline architecture: User Prompt -> Context Retrieval -> Prompt Construction -> LLM Synthesis.
