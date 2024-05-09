# Disaster_response_web_app
A web application for classifiy disaster related message. There are 36 categories. 

### Description
* Input for the web: Any message.

* Output from the Web App: A classification for the message if the message can be classified into 36 categories.

### Introduction to different parts of the project
1. Data

The data come from disaster data from [Appen](https://www.appen.com/). It contains the messages and its categories.
The data folder contains the processing code to clean the dataset.

2. Model

The model is a NLP pipeline and the classifier is 'RandomForestClassifier'

3. App

The app contains the files to deploy the web App.

### Instruction to run the Web App

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

Note: You need to have a localhost or a website to deploy the project.