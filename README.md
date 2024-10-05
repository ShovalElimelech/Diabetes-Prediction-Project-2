# Diabetes-Prediction-Project

In order to run the diabetes prediction application, which is based on deep learning models, the following actions are required:
--------------------

0. Preliminary preparations:
*At every stage, in order to do this in the most correct way for our computer's operating system, we will use Google/ChatGPT.

We will do all of this in a terminal window that we open inside the *primary project folder and execute the following commands:

0.1. We will create a virtual Python workspace for the project.
>>> python -m venv venv

>>> source venv/bin/activate (#note: for Linux/MacOS)

>>> venv\Scripts\activate (#note: for Windows)

0.2. We will download the required Python libraries into the virtual work environment that we created in the previous step.
The required libraries are listed in the "requirements.txt" file.
>>> pip install -r requirements.txt
--------------------

1. We have finished with the preliminary preparations of the project and now we will work to run the project:

1.1. We will run the model training code "Final_Project_Code_models_part.py".
>>> python Code_Files/Final_Project_Code_app_part.py

1.2. We will run the application code "Final_Project_Code_app_part.py".
>>> streamlit run Code_Files/Final_Project_Code_app_part.py
