# Binary classification problem
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import pandas
import matplotlib.pyplot as plt 

tf.logging.set_verbosity(tf.logging.INFO)

def fit(diabetes_data):
    # Creating out feature columns for tensorflow , we need to pass a list of featur column objects 

    # All of these continuous values go in numeric columns
    num_preg = tf.feature_column.numeric_column('Number_pregnant')
    plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
    blood_press = tf.feature_column.numeric_column('Blood_pressure')
    Triceps = tf.feature_column.numeric_column('Triceps')
    insulin = tf.feature_column.numeric_column('Insulin')
    bmi = tf.feature_column.numeric_column('BMI')
    diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
    age = tf.feature_column.numeric_column('Age')

    # The categorical features can be encoded using a vocabulary list or a hash-bucket 
    # Vocabulary list- I give the string name of the column in my dataframe and the possible values I can expect there
    assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

    # But mostly we cannot possibly write out the names of all the country name of the world, so in that case we will use hash-bucket technique 
    # hash bucket size is maximum amount of categories that I can expect. It is ok to have hash-bucket size larger then actual categories
    assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

    # Now we can also convert a continuous column into a categorical column 
    diabetes_data['Age'].hist(bins=20)
    plt.show()

    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30 , 40 , 50 , 60 , 70 , 80 ])

    feature_columns = [num_preg , plasma_gluc,blood_press , Triceps , insulin , bmi , diabetes_pedigree , age_bucket , assigned_group]


    # We now perform a train test split
    Xdata = diabetes_data.drop('Class', axis=1) 
    labels = diabetes_data['Class']
    
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(Xdata , labels,test_size=0.3, random_state=101)

    # Once we have sorted our data , initialized and created a feature colum  we will create an input_function
    input_func = tf.estimator.inputs.pandas_input_fn(x=Xtrain,y=Ytrain,batch_size=10 , num_epochs=1000,shuffle=True)

    # Now lets create a model 
    model = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2)

    model.train(input_fn=input_func, steps=1000)

    # Evaluating our model 

    eval_input_func = tf.estimator.inputs.pandas_input_fn(x=Xtest,y=Ytest,batch_size=10,num_epochs=1, shuffle=False)

    results=model.evaluate(eval_input_func)

    predict_input_func = tf.estimator.inputs.pandas_input_fn(x=Xtest,batch_size=10, num_epochs=1 , shuffle=False)
    predictions = model.predict(predict_input_func)

    predictions = list(predictions)

    ###################################################################################################

    

    #model_dense.train(input_fn=input_func,steps=1000) # This gives an error 
    # Simply passing a feature column to a dense neural network gives an error, so we should wrap the categorical columns
    # with an embedding_column or a indicator_column. We can covert categorical columms as follows 

    embedded_group_columns = tf.feature_column.embedding_column(assigned_group, dimension=4)

    # Reinitialializing our feature columns
    feature_columns = [num_preg , plasma_gluc,blood_press , Triceps , insulin , bmi , diabetes_pedigree , age_bucket , embedded_group_columns]

    # Training a dense neural_network model a.k.a ANN

    model_dense = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,10,10], n_classes=2)

    model_dense.train(input_fn=input_func, steps=1000)



def main():
    # Get the data 
    path = './data/pima-indians-diabetes.csv'
    diabetes_data = pandas.read_csv(path)

    data_columns=diabetes_data.columns

    # Normalizing our columns / Data 
    print(diabetes_data.head())
    
    cols_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
    

    # Lambda expression with Pandas
    diabetes_data[cols_to_normalize] = diabetes_data[cols_to_normalize].apply(lambda x: (x-x.min())/(x.max()-x.min()))

    # Now for the estimator API we need to create the feature column and numeric columns of these present columns 



    fit(diabetes_data)


if __name__=='__main__':
    main()