🖥️ How to Launch the Interactive GUI

Follow these steps to set up and launch the Streamlit graphical user interface:
1. Prerequisites

    Python: Ensure Python 3.8+ is installed on your system. Verify by running python --version in your terminal.

    Working Directory: Open your Terminal, PowerShell, or Command Prompt and ensure you are in the root directory of the project.

2. Install Dependencies

Install all required project libraries, or Streamlit individually:
Bash

# Recommended: Install all project dependencies at once
pip install -r requirements.txt

# Alternatively: Install Streamlit only
pip install streamlit

3. Run the GUI Command

Execute the GUI script using Python's module launcher:
Bash

python -m streamlit run src/GUI.py

(Or run streamlit run src/GUI.py directly if Streamlit is added to your system PATH)
4. Access the Interface

Once executed, Streamlit will automatically open the application in your default web browser. If it does not open automatically, copy and paste the local network URL displayed in your terminal:

    Local URL: http://localhost:8501

💡 Troubleshooting Tips:

    ModuleNotFoundError: Run pip install -r requirements.txt to ensure all supporting data processing libraries (pandas, scikit-learn, joblib) are present.

    File Path Errors: If terminal outputs File does not exist: src/GUI.py, verify that your terminal path is at the project root (Obesity/) and not inside src/.