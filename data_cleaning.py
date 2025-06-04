# iberaries
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder , MinMaxScaler
import numpy as np 
import matplotlib.pyplot as plt

#  importing the data 
dataset = pd .read_csv('healthcare-dataset-stroke-data.csv')
df = pd.DataFrame(dataset )
print ( f"The original data overview : \n  {df.head()}")

#let's check for each column 
#********************** gender column ********************************
df['gender'] = df['gender'].replace("Other" , df['gender'].mode()[0]) #replace the type other by the mode value 
gender_encoder = LabelEncoder()
df['gender']= gender_encoder.fit_transform(df['gender'])# encoding the values into 0 for female and 1 for male 


# ********************** age column *******************************
# on this column there is no null values but there is inconsistant data like 0.765 and this can't be on age let's deal with this 
df['age'] = df['age'].astype(int)
mean_ages  = df[(df['age']>=1)&(df['age']<=100)]['age'].mean()
df['age'] = df['age'].apply(lambda x : mean_ages if x <=1 or x >=100 else x )
age_scaller = StandardScaler() # normalizing the values of age for model training using standarscaller for better dealing with outliers 
df['age'] = age_scaller.fit_transform(df['age'].values.reshape(-1,1)) # reshape converting from 1D to 2D 


# ********************ever_married column ***************************
df['ever_married']=LabelEncoder().fit_transform(df['ever_married']) # encoding the column into 0 for No and 1 for Yes 

# *********************** work_type column ****************************
# threr is 5 categories {Private, Self-employed, children, Govt_job, Never_worked} 
work_encoder = LabelEncoder()
df['work_type'] = work_encoder.fit_transform(df['work_type'])
work_list_encoded = dict(zip(work_encoder.classes_ , work_encoder.transform(work_encoder.classes_)))
# print (work_list_encoded)

#  ************************* Residence_type column ***********************
df['Residence_type']=LabelEncoder().fit_transform(df['Residence_type']) #encode to 1 for Urban and 0 for Rural  

# ********************** avg_glucose_level column ************************

'''
there is no null values but there is some inconsistant value so i searched for the suitable range of glucose 
levels it's between 65  and 190  so let's deal with this... 

'''
def remove_outliers (series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    IQR = q3 - q1 
    lower_band = q1 -1.5 * IQR 
    upper_band = q3 + 1.5 * IQR 
    return series.where ((series >= lower_band ) & ( series <= upper_band), np.nan)

df [ 'avg_glucose_level'] = remove_outliers(df['avg_glucose_level'])#this cause a 627 null values
df ['avg_glucose_level'] =df['avg_glucose_level'].fillna(df['avg_glucose_level'].mean() )
avg_g_level_scaller = MinMaxScaler()
df['avg_glucose_level'] = avg_g_level_scaller.fit_transform(df['avg_glucose_level'].values.reshape(-1,1))# nomalizing the values for ml 

#  ************************** bmi column **************************
# print ( df .isnull().sum ()) # tere is 201 null values 
df ['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['bmi'] = StandardScaler().fit_transform(df['bmi'].values.reshape(-1,1))

# ************************** smoking_status ***********************
# there is no null value but there is a category called unknown which means unavailable data so i will replace it with the mode value 
df['smoking_status'] = df['smoking_status'].replace("Unknown",df['smoking_status'].mode()[0] ) 
smoking_encoder = LabelEncoder()
df['smoking_status'] = smoking_encoder.fit_transform(df['smoking_status'])
smoking_encoder_list = dict(zip(smoking_encoder.classes_ ,smoking_encoder.transform(smoking_encoder.classes_)))

# saving the cleaned data 
df.to_csv("my_last_data_cleaned.csv",index=False)

