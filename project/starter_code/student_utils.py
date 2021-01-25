import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    code_map = pd.Series(ndc_df['Non-proprietary Name'].values,index=ndc_df.NDC_Code).to_dict()
    df['generic_drug_name'] = df.ndc_code.map(code_map)

    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df = df.sort_values('encounter_id')
    first_encounter_values = df.groupby('patient_nbr')['encounter_id'].head(1).values
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_values)]
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', train_fraction = 0.6, validation_fraction = 0.2):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    assert train_fraction + validation_fraction < 1

    df_rand = df.sample(frac=1)
    patient_ids_rand = df_rand[patient_key].unique()
    split_index_train = int(len(patient_ids_rand) * train_fraction)
    split_index_val = int(len(patient_ids_rand) * (train_fraction + validation_fraction))
    patient_ids_train = patient_ids_rand[:split_index_train]
    patient_ids_validation = patient_ids_rand[split_index_train:split_index_val]
    patient_ids_test = patient_ids_rand[split_index_val:]

    train = df_rand[df_rand[patient_key].isin(patient_ids_train)]
    validation = df_rand[df_rand[patient_key].isin(patient_ids_validation)]
    test = df_rand[df_rand[patient_key].isin(patient_ids_test)]

    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        #embedding_size = round(tf_categorical_vocab.num_buckets ** 0.25)
        #tf_categorical_feature_column = tf.feature_column.embedding_column(tf_categorical_vocab, embedding_size)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_vocab)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    return tf.feature_column.numeric_column(
        key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = np.squeeze(diabetes_yhat.mean())
    s = np.squeeze(diabetes_yhat.stddev())
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >=5 else 0).values
    return student_binary_prediction
