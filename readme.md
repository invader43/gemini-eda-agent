# AI-Powered EDA Chat Agent

This repository contains a single-file implementation of an AI-Powered Exploratory Data Analysis (EDA) Chat Agent. The agent is built using LangGraph and Chainlit, with Google Gemini as the core language model. It allows users to upload a CSV file, ask questions about the data, and receive insights, including automatically generated charts.

This project is designed to be a Minimum Viable Product (MVP) demonstrating how to create an autonomous agent that can plan, execute code, and reflect on the results to perform data analysis tasks in a conversational manner.

## Features

-   **Conversational EDA**: Interact with your data using natural language.
-   **CSV Upload**: Easily upload your own datasets for analysis.
-   **Autonomous Agent Loop**: The agent uses a Plan-Code-Reflect loop to break down and solve data analysis problems.
-   **Dynamic Chart Generation**: Automatically generates and displays relevant charts (histograms, scatter plots, etc.) to visualize the data.
-   **Step-by-Step Progress**: The Chainlit UI provides real-time feedback on the agent's current task (Planning, Coding, Reflecting).
-   **Comprehensive Logging**: Detailed logs are generated for each session, capturing the agent's actions, generated code, and outputs for easy debugging and review.
-   **Single-File Implementation**: The entire application is contained in a single Python script for simplicity and ease of understanding.

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Python 3.8+**
-   **uv**: A fast, next-generation Python package manager. This project uses `uv` for dependency management instead of `pip` and `venv`.

### Installing uv

If you don't have `uv` installed, you can install it with a single command.

**On macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"```

You can verify the installation by running:
```bash
uv --version
```

## Installation

Follow these steps to set up and run the project using `uv`.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/ai-eda-chat-agent.git
    cd ai-eda-chat-agent
    ```

2.  **Create a Virtual Environment:**
    Use `uv` to create a new virtual environment. This will create a `.venv` directory in your project folder.
    ```bash
    uv venv
    ```

3.  **Activate the Virtual Environment:**

    **On macOS and Linux:**
    ```bash
    source .venv/bin/activate
    ```

    **On Windows:**
    ```powershell
    .venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    The required packages are listed in the `requirements.txt` file. Use `uv` to install them. `uv` is significantly faster than `pip`.
    ```bash
    uv pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain the following packages:
    ```
    pandas
    matplotlib
    seaborn
    chainlit
    langchain-google-genai
    langgraph
    python-dotenv
    ```

## Configuration

The agent uses the Google Gemini API. You need to provide an API key for the application to work.

1.  **Create a `.env` file** in the root of the project directory.

2.  **Add your Google API key** to the `.env` file:
    ```
    GOOGLE_API_KEY="your-google-api-key-here"
    ```

    You can obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## How to Run the Application

Once you have completed the installation and configuration steps, you can start the Chainlit application.

1.  **Ensure your virtual environment is activated.**

2.  **Run the application using the Chainlit CLI:**
    ```bash
    chainlit run app.py -w
    ```
    The `-w` flag enables auto-reloading, which is useful for development.

3.  **Open your web browser** and navigate to `http://localhost:8000`.

4.  You will be prompted to upload a CSV file. Once uploaded, you can start asking questions about your data.

## Project Structure

The project is contained within a single file, `app.py`, which is organized as follows:

1.  **Imports and Setup**: Imports necessary libraries and configures logging, Matplotlib/Seaborn styles.
2.  **Logging Configuration**: Sets up a robust logging system that saves detailed logs to a file while showing a cleaner version in the console.
3.  **Agent State Definition**: Defines the `AgentState` TypedDict, which manages the flow of data through the LangGraph agent.
4.  **LangGraph Nodes**:
    *   `planner_node`: Analyzes the user's question and history to decide the next step.
    *   `coder_node`: Generates and executes Python code for data analysis and visualization.
    *   `reflector_node`: Examines the output of the code to determine if the question has been answered or if more steps are needed.
5.  **Graph Definition**:
    *   `should_continue`: A conditional edge that determines whether the agent's work is complete or if it should continue to the next iteration.
    *   The nodes and edges are compiled into the `eda_agent` graph.
6.  **Chainlit Application**:
    *   `@cl.on_chat_start`: Handles the initial user interaction, including file upload and validation.
    *   `@cl.on_message`: The main function that gets triggered for each user message. It invokes the LangGraph agent and streams the progress back to the user, displaying the final answer and any generated charts.
