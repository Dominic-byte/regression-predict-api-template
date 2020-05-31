"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# Feature engineering
import geopy.distance

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')

def data_cleaner(df, dropdiff = False):
    df1 = df.copy()
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
    df1 = col_rename(df1)

    if dropdiff == True:
        for col in atrain_df.columns:
            if col not in atest_df.columns and col != 'Time_from_Pickup_to_Arrival':
                print(col)
                atrain_df.drop(columns=col, inplace = True)

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
        # Create lists of columns to change
        integers = [col for col in cdf.columns if cdf[col].dtypes == 'int64']
        floats = [col for col in cdf.columns if cdf[col].dtypes == 'float64']
        time_cols = [col for col in cdf.columns if col.endswith("Time")]

        # Reduce size of data storage types
        cdf[integers] = cdf[integers].astype('int16')
        cdf[floats] = cdf[floats].astype('float16')
        for col in time_cols:
            cdf[col] = pd.to_datetime(cdf[col])
            cdf[col] = [time.time() for time in cdf[col]]
            cdf[col] = cdf[col].apply(lambda x: x.hour)
        # Correcting specific columns
        if 'Distance (KM)' in cdf.columns:
            cdf['Distance (KM)'] = cdf['Distance (KM)'].astype('float16') 

        new = cdf.memory_usage().sum()
        return cdf
    df1 = data_preprocessing(df1)
    df2 = col_rename(riders)
    df2 = data_preprocessing(df2)
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
        col_df = pd.merge(df, df2, how='left', on='Rider_Id')
        droplist = ['Place_DoM', 'Place_Weekday',
                    'Place_Time', 'Confirm_DoM', 
                    'Confirm_Weekday', 'Confirm_Time',
                    'Arr_Pickup_DoM', 'Arr_Pickup_Weekday',
                    'Arr_Pickup_Time']
        if droplist[0] in col_df.columns:
            col_df = col_df.drop(columns=droplist)

        return col_df
    df1 = collinear(df1)


    def null(df, na_thresh = 1.0 , strategy = "median"):
        """Function that drops columns with more than na_thresh null from

        """
        no_null=df.copy()
        # Dropping NaN's
        for col in no_null.columns:
            missing = no_null[col].isnull().sum()/len(no_null)
            if missing > na_thresh:
                no_null.drop(columns=col, inplace=True)

        # Filling NaN's
        for col in no_null.columns:
            if no_null[col].isnull().sum() > 0:
                if no_null[col].dtypes == 'object':
                    no_null[col].fillna(no_null[col].mode(), inplace=True) 
                elif strategy == 'mean':
                    no_null[col].fillna(round(no_null[col].mean()), inplace=True)
                elif strategy == 'median':
                    no_null[col].fillna(round(no_null[col].median()), inplace=True)
                elif strategy == 'rolling':
                    no_null[col].fillna(no_null[col].rolling(7).mean(), inplace=True)
                else:
                    raise ValueError
        return no_null
    
    df1  = null(df1, na_thresh = 1.0 , strategy = "median")

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
        droplist = ['User_Id', 'Vehicle_Type', 'Rider_Id'] #, 'Pickup_Lat',	'Pickup_Long',	'Destination_Lat',	'Destination_Long', 'Temperature','Precipitation_mm', 'Pay_Day', 'Pickup_DoM']
        for col in droplist:
            if col in trans_df.columns:
                trans_df.drop(columns=col, inplace=True)

        # Reorder columns
        reindex_cols = [col for col in trans_df.columns if col != 'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']
        trans_df = trans_df.reindex(columns = reindex_cols)
        trans_df.index = (index)
        original_features = [col for col in trans_df.columns if col != 'Time_from_Pickup_to_Arrival']
        return trans_df, original_features
    df1, org_features = variable_transformer(df1)

    def variable_creator(df):
        """Doctstring here
        """
        create_df = df.copy()

        # covert GPS corrdinates to distance
        def dist(df,lat1,long1,lat2,long2):
            list1=[]
            geo_df = df.copy()
            for i in range (0,len(geo_df)):
                coords_1 = (geo_df[lat1][i], geo_df[long1][i])
                coords_2 = (geo_df[lat2][i], geo_df[long2][i])

                distance=geopy.distance.vincenty(coords_1, coords_2).m
                #print(distance)
                list1.append(distance)
            geo_df['Geo_Distance'] = list1
            return geo_df
        create_df = dist(create_df,'Pickup_Lat', 'Pickup_Long', 'Destination_Lat',
            'Destination_Long')

        # # Extract hours from columns
        # create_df['Pickup_Hour'] = pd.to_datetime(create_df['Pickup_Time'].astype(str)).dt.hour



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
        def comfort(temp):
            if temp < 18:
                return 0
            elif temp < 28.75:
                return 1
            return 2
        create_df['Comfort'] = create_df['Temperature'].apply(comfort)

        # Is pay day?
        def pay_day(DoM):
            if DoM in [30,31]:
                return True
            return False
        create_df['Pay_Day'] = create_df['Pickup_DoM'].apply(pay_day)

        # Get dummies
        dummy_cols = ['Pers_Business','Pickup_Weekday', 'Pickup_DoM','Platform','Pickup_Time']
        create_df = pd.get_dummies(create_df,columns = dummy_cols, drop_first=True,)

        # Reorder columns
        reindex_cols = [col for col in create_df.columns if col != 'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']
        create_df = create_df.reindex(columns = reindex_cols)



        create_features = [col for col in create_df.columns if col not in df.columns]

        return create_df, create_features
    
    df1, new_features = variable_creator(df1)

    return df1, org_features, new_features

train, org_features, new_features = data_cleaner(train, dropdiff = False)

y_train = train[['Time_from_Pickup_to_Arrival']]
X_train = train[['Distance_KM','Average_Rating', 'Driver_exp','Pers_Business_Personal',
                'Pickup_Weekday_2','Pickup_Weekday_5','Pickup_Weekday_6','Pickup_DoM_8',
                'Pickup_DoM_9','Pickup_DoM_11', 'Pickup_DoM_17','Pickup_DoM_22','Pickup_DoM_27',
                'Pickup_DoM_29','Pickup_DoM_30',  'Pickup_Time_8','Pickup_Time_9','Pickup_Time_10',
                'Pickup_Time_12','Pickup_Time_15','Pickup_Time_16']]


# Fit model
lgbr = lgb.LGBMRegressor(bagging_fraction = 0.1,
                         boosting_type='gbdt',
  feature_fraction = 0.2812379369053784,
  learning_rate = 0.32,
  max_bin = 49,
  max_depth = 4,
  min_data_in_leaf = 20,
  min_sum_hessian_in_leaf = 35.74236812873617,
  num_leaves = 75,
  subsample = 0.4615201756036703)
print ("Training Model...")
lgbr.fit(X_train, y_train)

# Instantiate BaggingRegressor model with a light gradient boosting regression as the base model
bag_reg = BaggingRegressor(base_estimator = lgbr)
bag_reg.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/sendy_lgb_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(bag_reg, open(save_path,'wb'))
