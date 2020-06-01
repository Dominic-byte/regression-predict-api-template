#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Dominic-byte/regression-predict-api-template/blob/master/model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[8]:


"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""


# In[9]:


# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import geopy


# In[10]:


test = pd.read_csv('utils/data/test_data.csv')
riders = pd.read_csv('utils/data/riders.csv')
test = test.merge(riders, how='left', on='Rider Id')

data = test.iloc[1].to_json()


# In[11]:


# Convert the json string to a python dictionary object
feature_vector_dict = json.loads(data)
# Load the dictionary as a Pandas DataFrame.
feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])


# In[12]:


feature_vector_df.iloc[0]['Order No']
query_num = test[test['Order No'] == feature_vector_df.iloc[0]['Order No']].index
test.iloc[query_num]


# In[50]:


riders = pd.read_csv('utils/data/riders.csv')

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    # Obtain query number
    test_df = test.copy()
    query_num = test_df[test_df['Order No'] == feature_vector_df.iloc[0]['Order No']].index 
    # ---------------------------------------------------------------
    # Rename Columns of test df
    test_df.iloc[query_num]

    
    def col_rename(df):
        """Function that renames columns and drops columns that 
        were not used in the test set
        Parameters
        ----------
        df: DataFrame
              input df to rename columns for
        Returns
        -------
        new_df: DataFrame
              output df with renamed and dropped columns
              as well as a printout of dropped columns
        """
        static_dict = {"Order No":"Order No","User Id":"User Id","Vehicle Type":"Vehicle Type","Platform Type":"Platform",
            "Personal or Business":"Pers Business","Placement - Day of Month":"Place DoM",
            "Placement - Weekday (Mo = 1)":"Place Weekday","Placement - Time":"Place Time",
            "Confirmation - Day of Month":"Confirm DoM","Confirmation - Weekday (Mo = 1)":"Confirm Weekday",
            "Confirmation - Time":"Confirm Time","Arrival at Pickup - Day of Month":"Arr Pickup DoM",
            "Arrival at Pickup - Weekday (Mo = 1)":"Arr Pickup Weekday","Arrival at Pickup - Time":"Arr Pickup Time",
            "Pickup - Day of Month":"Pickup DoM","Pickup - Weekday (Mo = 1)":"Pickup Weekday","Pickup - Time":"Pickup Time",
            "Distance (KM)":"Distance KM","Temperature":"Temperature","Precipitation in millimeters":"Precipitation mm",
            "Pickup Lat":"Pickup Lat","Pickup Long":"Pickup Long","Destination Lat":"Destination Lat","Time from Pickup to Arrival":"Time from Pickup to Arrival",
            "Destination Long":"Destination Long","Rider Id":"Rider Id"}
        new_df = df.copy()
        new_cols = {}
        droplist = []
        for col in new_df.columns:
            if col in static_dict.keys():
                new_cols[col] = static_dict[col].replace(" ","_")
        else:
            droplist.append(col)
        new_df.rename(columns = new_cols, inplace=True)
        for col in droplist:
            if col in new_df.index:
                return new_df.drop(columns=droplist, inplace=True)
            return new_df
    
    test_df = col_rename(test_df)
        
    def data_preprocessing(df):
        """Function that preprocesses data used for predictions and testing
        Parameters
        ----------
        df: DataFrame
              DataFrame to preprocess.
        Returns
        -------
        df: DataFrame
              Preprocessed DataFrame

        """
        cdf = df.copy()
        old = cdf.memory_usage().sum()
        
        # Convert time columns to hour
        time_cols = [col for col in cdf.columns if col.endswith("Time")]
        for col in time_cols:
            cdf[col] = pd.to_datetime(cdf[col])
            cdf[col] = [time.time() for time in cdf[col]]
            cdf[col] = cdf[col].apply(lambda x: x.hour)
            
        # Create lists of columns to change
        integers = [col for col in cdf.columns if cdf[col].dtypes == 'int64']
        floats = [col for col in cdf.columns if cdf[col].dtypes == 'float64']
        
        # Reduce size of data storage types
        cdf[integers] = cdf[integers].astype('int16')
        cdf[floats] = cdf[floats].astype('float16')
        # Correcting specific columns
        if 'Distance (KM)' in cdf.columns:
            cdf['Distance (KM)'] = cdf['Distance (KM)'].astype('float16') 

        new = cdf.memory_usage().sum()
        
        return cdf
    
    test_df = data_preprocessing(test_df)
   
    
    def collinear(df):
        """Function that drops chosen columns
        Parameters
        ----------
        df: DataFrame
              DataFrame to drop columns from
        cols: list-like
              Names of columns to be dropped from
        Returns
        -------
        col_df: DataFrame
              Modified DataFrame
        """
        col_df = df.copy()
        droplist = ['Place_DoM', 'Place_Weekday',
                  'Place_Time', 'Confirm_DoM', 
                  'Confirm_Weekday', 'Confirm_Time',
                  'Arr_Pickup_DoM', 'Arr_Pickup_Weekday',
                  'Arr_Pickup_Time']
        if droplist[0] in col_df.columns:
            col_df = col_df.drop(columns=droplist)

        return col_df

    test_df = collinear(test_df)

#     def null(df, na_thresh = 1.0 , strategy = "median"):
#         """Function that drops columns with more than na_thresh null from

#         """
#         no_null=df.copy()
#         # Dropping NaN's
#         for col in no_null.columns:
#             missing = no_null[col].isnull().sum()/len(no_null)
#         if missing > na_thresh:
#             no_null.drop(columns=col, inplace=True)

#         # Filling NaN's
#         for col in no_null.columns:
#             if no_null[col].isnull().sum() > 0:
#                 if no_null[col].dtypes == 'object':
#                     no_null[col].fillna(no_null[col].mode(), inplace=True) 
#                 elif strategy == 'mean':
#                     no_null[col].fillna(round(no_null[col].mean()), inplace=True)
#                 elif strategy == 'median':
#                     no_null[col].fillna(round(no_null[col].median()), inplace=True)
#                 elif strategy == 'rolling':
#                     no_null[col].fillna(no_null[col].rolling(7).mean(), inplace=True)
#                 else:
#                     raise ValueError
#         return no_null    

    def variable_transformer(df):
        """Function to apply variable ALL transformations to dataset
        Parameters
        ----------
        df: DataFrame
              Input df to be transformed
        Returns
        -------
        trans_df: DataFrame
              Output df
        """
        index = df.index
        trans_df = df.copy()

        # # Time to number
        # for time in time_cols:
        #   if time in trans_df.columns:
        #     trans_df[time] = pd.to_datetime(trans_df[time])



        # # Feature scaling
        # predictor_cols = ['Distance_KM'	,'Temperature	','Precipitation_mm', 'No_Of_Orders', 'Age', 'Average_Rating',	'No_of_Ratings']
        # sc = StandardScaler()
        # sc.fit_transform(trans_df[predictor_cols])

        # Drop unnecessary features
        droplist = ['User_Id', 'Vehicle_Type', 'Rider_Id', 'Temperature', 'Precipitation_mm'] #, 'Pickup_Lat',	'Pickup_Long',	'Destination_Lat',	'Destination_Long', 'Temperature','Precipitation_mm', 'Pay_Day', 'Pickup_DoM']
        for col in droplist:
            if col in trans_df.columns:
                trans_df.drop(columns=col, inplace=True)

        # Reorder columns
        reindex_cols = [col for col in trans_df.columns if col != 'Time_from_Pickup_to_Arrival']
        trans_df = trans_df.reindex(columns = reindex_cols)
        trans_df.index = (index)
        original_features = [col for col in trans_df.columns if col != 'Time_from_Pickup_to_Arrival']
        return trans_df, original_features    

    test_df, ori_feats = variable_transformer(test_df)

    def variable_creator(df):
        """Doctstring here
        """
        create_df = df.copy()
        
        # covert GPS corrdinates to distance
#         def dist(df,lat1,long1,lat2,long2):
#             list1=[]
#             geo_df = df.copy()
#             for i in range (0,len(geo_df)):
#                 coords_1 = (geo_df[lat1][i], geo_df[long1][i])
#                 coords_2 = (geo_df[lat2][i], geo_df[long2][i])

#                 distance = geopy.distance.geodesic.vincenty(coords_1, coords_2).m

#                 list1.append(distance)
#             geo_df['Geo_Distance'] = list1
#             return geo_df
#         create_df = dist(create_df,'Pickup_Lat', 'Pickup_Long', 'Destination_Lat',
#             'Destination_Long')

        # Extract hours from columns
#         create_df['Pickup_Hour'] = pd.to_datetime(create_df['Pickup_Time'].astype(str)).dt.hour



        # Determine which platform is busiest
        def is_busy(platform):
            if platform == 3:
                return True
            else:
                return False
        create_df['Platform_load'] = create_df['Platform'].apply(is_busy)
        
        # Which drivers are the most experienced
        def experience(age):
            if age < 1000:
                return 1
            elif age < 2000:
                return 2
            elif age < 3000:
                return 3
            return 4
        create_df['Driver_exp'] = create_df['Age'].apply(experience)
        
        # Business day
        def weekday(day):
            if day in [1,2,3,4,5]:
                return 1
            return 0
        create_df['Business_day'] = create_df['Pickup_Weekday'].apply(weekday)
        
        # Is it hot?
#         def comfort(temp):

#             if temp < 18:
#                 return 0
#             elif temp < 28.75:
#                 return 1
#             return 2
#         create_df['Comfort'] = create_df['Temperature'].apply(comfort)

        # Is pay day?
        def pay_day(DoM):
            if DoM in [30,31]:
                return True
            return False
        
        create_df['Pay_Day'] = create_df['Pickup_DoM'].apply(pay_day)
       
        
       
        
        
        
        # Reorder columns
        reindex_cols = [col for col in create_df.columns if col != 'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']
        create_df = create_df.reindex(columns = reindex_cols)

        create_features = [col for col in create_df.columns if col not in df.columns]

        return create_df, create_features
    test_df, new_feats = variable_creator(test_df) 
   
    
    def get_dum(df):
        # Get dummies
        dummy_df = df.copy()
        dummy_cols = ['Pers_Business','Pickup_Weekday','Pickup_DoM','Platform','Pickup_Time']
        dummy_df = pd.get_dummies(dummy_df,columns=dummy_cols,drop_first=True)
        return dummy_df
    
    test_df = get_dum(test_df)
      
    test_df = test_df.iloc[query_num]
   
    predictor_feats = [# 'Feature',  # Individual P-Score
                  # 'Platform',
                  # 'Pers_Business',
                  # 'Pickup_DoM',
                  # 'Pickup_Weekday',
                  # 'Pickup_Time',
                  'Distance_KM', # P - 0.00
                  # 'Temperature', # P - 0.305
                  # 'Precipitation_mm', # P - 0.216
                  # 'Pickup_Lat', # location does not consider obstacles
                  # 'Pickup_Long',  # location does not consider obstacles
                  # 'Destination_Lat',  # location does not consider obstacles
                  # 'Destination_Long', # location does not consider obstacles 
                  # 'No_Of_Orders', # condition number is large, 5.09e+03.
                  # 'Age', # condition number is large, 2.88e+03.
                  'Average_Rating', # P - 0.000
                  # 'No_of_Ratings', # condition number is large, 1.11e+03.
                  # 'Geo_Distance',  # condition number is large, 1.6e+04.
                  # 'Platform_load', # P - 0.229
                  'Driver_exp', # P - 0.000
                  # 'Business_day', # P - 0.000 # multicollinearity
                  # 'Comfort', # P - 0.104
                  # 'Pay_Day', # P - 0.306
                  'Pers_Business_Personal', # P - 0.002
                  'Pickup_Weekday_2', # P - 0.000
                  # 'Pickup_Weekday_3', # P - 0.369
                  # 'Pickup_Weekday_4', # P - 0.123
                  'Pickup_Weekday_5', # P - 0.000 
                  'Pickup_Weekday_6', # P - 0.000
                  # 'Pickup_Weekday_7', # P - 0.292
                  # 'Pickup_DoM_2', # P - 0.939
                  # 'Pickup_DoM_3', # P - 0.947
                  # 'Pickup_DoM_4', # P - 0.355
                  # 'Pickup_DoM_5', # P - 0.577
                  # 'Pickup_DoM_6', # P - 0.737
                  # 'Pickup_DoM_7', # P - 0.304
                  'Pickup_DoM_8', # P - 0.023
                  'Pickup_DoM_9', # P - 0.021
                  # 'Pickup_DoM_10', # P - 0.725
                  'Pickup_DoM_11', # P - 0.140
                  # 'Pickup_DoM_12', # P - 0.332
                  # 'Pickup_DoM_13', # P - 0.550
                  # 'Pickup_DoM_14', # P - 0.288
                  # 'Pickup_DoM_15', # P - 0.735
                  # 'Pickup_DoM_16', # P - 0.195
                  'Pickup_DoM_17', # P - 0.098
                  # 'Pickup_DoM_18', # P - 0.229
                  # 'Pickup_DoM_19', # P - 0.425
                  # 'Pickup_DoM_20', # P - 0.845
                  # 'Pickup_DoM_21', # P - 0.553
                  'Pickup_DoM_22', # P - 0.098
                  # 'Pickup_DoM_23', # P - 0.500
                  # 'Pickup_DoM_24', # P - 0.627
                  # 'Pickup_DoM_25', # P - 0.352
                  # 'Pickup_DoM_26', # P - 0.245
                  'Pickup_DoM_27', # P - 0.001
                  # 'Pickup_DoM_28', # P - 0.746
                  'Pickup_DoM_29', # P - 0.096
                  'Pickup_DoM_30', # P - 0.079 
                  # 'Pickup_DoM_31', # P - 0.565
                  # 'Platform_2', # P - 0.342
                  # 'Platform_3', # P - 0.229
                  # 'Platform_4', # P - 0.155
                  # 'Pickup_Time_7',  # P - 0.947
                  'Pickup_Time_8',  # P - 0.043
                  'Pickup_Time_9',  # P - 0.000
                  'Pickup_Time_10', # P - 0.000
                  # 'Pickup_Time_11', # P - 0.380
                  'Pickup_Time_12', # P - 0.086
                  # 'Pickup_Time_13', # P - 0.879
                  # 'Pickup_Time_14', # P - 0.625
                  'Pickup_Time_15', # P - 0.000
                  'Pickup_Time_16', # P - 0.086
                  # 'Pickup_Time_17', # P -0.325
                  # 'Pickup_Time_18', # P - 0.585
                  # 'Pickup_Time_19', # P - 0.741
                  # 'Pickup_Time_20', # P - 0.461
                  # 'Pickup_Time_21', # P - 0.300
                  # 'Pickup_Time_22', # P - 0.158
                  # 'Pickup_Time_23' # P - 0.514
    ]
    # ----------- Replace this code with your own preprocessing steps --------
  
    predict_vector = test_df[predictor_feats]
    # ------------------------------------------------------------------------

    return predict_vector


# In[51]:


predict_vector = _preprocess_data(data)


# In[52]:


predict_vector.head()


# In[57]:


def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


# In[58]:


def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    print(prep_data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()





