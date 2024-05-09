import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine, text

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    INPUT:
    database_filepath - the filepath for my database 
    
    OUTPUT:
    X - message column from dataset
    y - catogory for the messages (labels)
    category_names - category(label) names for y
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    # Define your SQL query
    sql_query = text("SELECT * FROM data_cleaned")

    # Load data into a DataFrame using pandas' read_sql function

    df = pd.read_sql_query(sql_query, engine.connect())
    
    X = df.message.values
    y = df.iloc[:, 4:]
    category = y.columns
    
    return X, y, category


def tokenize(text):
    """
    INPUT:
    text - the messages from the dataset
    
    OUTPUT:
    clean_tokens - tokenized text for the message from dataset
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    INPUT: None
    
    OUTPUT:
    cv = ML model pipeline with grid search parameter set up
    """
    # Define classifier
    classifier = RandomForestClassifier()

    # Create a MultiOutputClassifier with the base classifier
    multi_output_classifier = MultiOutputClassifier(classifier)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',multi_output_classifier)
    ])
    parameters = {
    #'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__min_samples_split': [2]
        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    

        
def display_cv_results(cv, Y_test, Y_pred):
    """
    INPUT: 
    cv = ML model pipeline with grid search parameter set up
       
    OUTPUT:
    none 
    
    FUNCTION:
    print confusion matirx, accuracy and best parameter from grid         search
    """
    print("\nBest Parameters:", cv.best_params_)
    for i,column in enumerate(Y_test.columns):  # Assuming y_true and y_pred are DataFrames
        labels = np.unique(Y_pred[:,i])
        confusion_mat = confusion_matrix(Y_test[column], Y_pred[:,i], labels=labels)
        accuracy = (Y_test[column] == Y_pred[:,i]).mean()
        
        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
        print('\n')


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test data set for X
    y_test - test data set for y
    category_names - category name for y
    
    OUTPUT:
    none
    
    FUNCTION:
    use the trained model to predict, and call the display function
    to display evaluation report for the model
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)

    # display results
    display_cv_results(model, Y_test, Y_pred)


def save_model(model, model_filepath):
    """
    INPUT:
    model - trained model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    
    FUNCTION:
    save the trained model to the given location
    """
    
    # Save the model to a file using pickle
    with open('multi_random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()