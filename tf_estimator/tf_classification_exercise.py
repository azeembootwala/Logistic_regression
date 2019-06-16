import numpy as np 
import tensorflow as tf
import pandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

tf.logging.set_verbosity(tf.logging.INFO)

def fit(Xtrain , Xtest , Ytrain , Ytest):
    age = tf.feature_column.numeric_column('age')
    workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
    education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
    education_num = tf.feature_column.numeric_column('education_num')
    marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
    relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)
    race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=1000)
    gender = tf.feature_column.categorical_column_with_hash_bucket('gender', hash_bucket_size=1000)
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)

    feature_columns = [age , workclass , education , education_num , 
                        marital_status, occupation , relationship , race , gender ,capital_gain , capital_loss, 
                        hours_per_week , native_country]
    
    # Now that feature columns are created , we move on with input_function 

    input_func = tf.estimator.inputs.pandas_input_fn(x=Xtrain , y=Ytrain , batch_size=10 , num_epochs=1000 , shuffle=True)

    prediction_input_func = tf.estimator.inputs.pandas_input_fn(x=Xtest , batch_size=10 , num_epochs=1 , shuffle=False)

    model = tf.estimator.LinearClassifier(feature_columns=feature_columns , n_classes=2)

    model.train(input_fn = input_func, steps= 1000)

    prediction = list(model.predict(input_fn = prediction_input_func))

    final_preds = []
    for pred in prediction:
        final_preds.append(pred['class_ids'][0])

    print(classification_report(final_preds , Ytest))


def main():
    path = './data/census_data.csv'
    Data = pandas.read_csv(path)
    # We will first normalize the continuous columns and before that we need to figure out which columns to normalize 
    Xdata = Data.drop('income_bracket', axis = 1)
    Ydata = Data['income_bracket']
    
    # Converting string values into numerical categories 
    Ydata = Ydata.apply(lambda x:0 if x==' <=50K' else 1)

    continuous_columns = ['age','education_num','capital_gain','capital_loss','hours_per_week']

    # Splitting our data into Train and Test set 
    Xtrain, Xtest , Ytrain , Ytest  = train_test_split(Xdata , Ydata , test_size=0.3 , random_state=101)

    Xtrain[continuous_columns] = Xtrain[continuous_columns].apply(lambda x:(x - x.min())/(x.max()-x.min()))
 
    Xtest[continuous_columns] = Xtest[continuous_columns].apply(lambda x:(x - x.min())/ (x.max() - x.min()))

    fit(Xtrain , Xtest , Ytrain , Ytest)
    

if __name__=='__main__':
    main()