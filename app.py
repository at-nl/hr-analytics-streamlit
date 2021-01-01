import os
import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import bz2
import _pickle as cPickle
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

##### Load model
encoder = pickle.load(open(r'encoder.pkl', 'rb'))
scaler_smote = pickle.load(open(r'scaler_smote.pkl', 'rb'))
pca_smote = pickle.load(open(r'pca_smote.pkl', 'rb'))
# best_clf = pickle.load(open(download_link, 'rb'))
best_clf = bz2.BZ2File(r'best_model_compressed.pbz2', 'rb')
best_clf = cPickle.load(best_clf)
colnames = ['city_city_1', 'city_city_10', 'city_city_100', 'city_city_101',
       'city_city_102', 'city_city_103', 'city_city_104', 'city_city_105',
       'city_city_106', 'city_city_107', 'city_city_109', 'city_city_11',
       'city_city_111', 'city_city_114', 'city_city_115', 'city_city_116',
       'city_city_117', 'city_city_118', 'city_city_12', 'city_city_120',
       'city_city_121', 'city_city_123', 'city_city_126', 'city_city_127',
       'city_city_128', 'city_city_129', 'city_city_13', 'city_city_131',
       'city_city_133', 'city_city_134', 'city_city_136', 'city_city_138',
       'city_city_139', 'city_city_14', 'city_city_140', 'city_city_141',
       'city_city_142', 'city_city_143', 'city_city_144', 'city_city_145',
       'city_city_146', 'city_city_149', 'city_city_150', 'city_city_152',
       'city_city_155', 'city_city_157', 'city_city_158', 'city_city_159',
       'city_city_16', 'city_city_160', 'city_city_162', 'city_city_165',
       'city_city_166', 'city_city_167', 'city_city_171', 'city_city_173',
       'city_city_175', 'city_city_176', 'city_city_179', 'city_city_18',
       'city_city_180', 'city_city_19', 'city_city_2', 'city_city_20',
       'city_city_21', 'city_city_23', 'city_city_24', 'city_city_25',
       'city_city_26', 'city_city_27', 'city_city_28', 'city_city_30',
       'city_city_31', 'city_city_33', 'city_city_36', 'city_city_37',
       'city_city_39', 'city_city_40', 'city_city_41', 'city_city_42',
       'city_city_43', 'city_city_44', 'city_city_45', 'city_city_46',
       'city_city_48', 'city_city_50', 'city_city_53', 'city_city_54',
       'city_city_55', 'city_city_57', 'city_city_59', 'city_city_61',
       'city_city_62', 'city_city_64', 'city_city_65', 'city_city_67',
       'city_city_69', 'city_city_7', 'city_city_70', 'city_city_71',
       'city_city_72', 'city_city_73', 'city_city_74', 'city_city_75',
       'city_city_76', 'city_city_77', 'city_city_78', 'city_city_79',
       'city_city_8', 'city_city_80', 'city_city_81', 'city_city_82',
       'city_city_83', 'city_city_84', 'city_city_89', 'city_city_9',
       'city_city_90', 'city_city_91', 'city_city_93', 'city_city_94',
       'city_city_97', 'city_city_98', 'city_city_99', 'gender_Female',
       'gender_Male', 'gender_Other', 'gender_nan',
       'relevent_experience_Has relevent experience',
       'relevent_experience_No relevent experience',
       'enrolled_university_Full time course',
       'enrolled_university_Part time course', 'enrolled_university_nan',
       'enrolled_university_no_enrollment', 'education_level_Graduate',
       'education_level_High School', 'education_level_Masters',
       'education_level_Phd', 'education_level_Primary School',
       'education_level_nan', 'major_discipline_Arts',
       'major_discipline_Business Degree', 'major_discipline_Humanities',
       'major_discipline_No Major', 'major_discipline_Other',
       'major_discipline_STEM', 'major_discipline_nan', 'experience_1',
       'experience_10', 'experience_11', 'experience_12', 'experience_13',
       'experience_14', 'experience_15', 'experience_16', 'experience_17',
       'experience_18', 'experience_19', 'experience_2', 'experience_20',
       'experience_3', 'experience_4', 'experience_5', 'experience_6',
       'experience_7', 'experience_8', 'experience_9', 'experience_<1',
       'experience_>20', 'experience_nan', 'company_size_10/49',
       'company_size_100-500', 'company_size_1000-4999',
       'company_size_10000+', 'company_size_50-99',
       'company_size_500-999', 'company_size_5000-9999',
       'company_size_<10', 'company_size_nan',
       'company_type_Early Stage Startup', 'company_type_Funded Startup',
       'company_type_NGO', 'company_type_Other',
       'company_type_Public Sector', 'company_type_Pvt Ltd',
       'company_type_nan', 'last_new_job_1', 'last_new_job_2',
       'last_new_job_3', 'last_new_job_4', 'last_new_job_>4',
       'last_new_job_nan', 'last_new_job_never']

##### Configure web app
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
st.set_option('deprecation.showImageFormat', False)
st.header("Employee Trajectory: Switch job vs Stay at current company")
image = Image.open(r'content\employees.jpg')
st.image(image, use_column_width = True, format = 'JPG')
st.sidebar.write("**Please insert values to determine whether an employee will stay at their current company or switch jobs.**")

##### Pull features
def format_cap(text):
  return text.replace('_',' ').strip().capitalize()
def format_city(city):
  return int(re.search('\d+',city).group())
def experience_st_func(num):
  if num < 1:
    return '<1'
  elif num > 20:
    return '>20'
  else:
    return (str(num))
def company_size_st_func(num):
  if num < 10:
    return '<10'
  elif num >=10 and num <= 49:
    return '10/49'
  elif num >=50 and num <= 99:
    return '50-99'
  elif num >=100 and num <= 499:
    return '100-500'
  elif num >=500 and num <= 999:
    return '500-999'
  elif num >=1000 and num <= 4999:
    return '1000-4999'
  elif num >=5000 and num <= 9999:
    return '5000-9999'
  else:
    return '10000+'
# City
# unique_cities = sorted([x for x in train['city'].unique().tolist() if str(x)!='nan'])
# unique_cities = sorted([int(re.search('\d+',x).group()) for x in unique_cities])
# unique_cities = ['city_{}'.format(i) for i in unique_cities]
unique_cities = ['city_1', 'city_2', 'city_7', 'city_8', 'city_9', 'city_10',
       'city_11', 'city_12', 'city_13', 'city_14', 'city_16', 'city_18',
       'city_19', 'city_20', 'city_21', 'city_23', 'city_24', 'city_25',
       'city_26', 'city_27', 'city_28', 'city_30', 'city_31', 'city_33',
       'city_36', 'city_37', 'city_39', 'city_40', 'city_41', 'city_42',
       'city_43', 'city_44', 'city_45', 'city_46', 'city_48', 'city_50',
       'city_53', 'city_54', 'city_55', 'city_57', 'city_59', 'city_61',
       'city_62', 'city_64', 'city_65', 'city_67', 'city_69', 'city_70',
       'city_71', 'city_72', 'city_73', 'city_74', 'city_75', 'city_76',
       'city_77', 'city_78', 'city_79', 'city_80', 'city_81', 'city_82',
       'city_83', 'city_84', 'city_89', 'city_90', 'city_91', 'city_93',
       'city_94', 'city_97', 'city_98', 'city_99', 'city_100', 'city_101',
       'city_102', 'city_103', 'city_104', 'city_105', 'city_106',
       'city_107', 'city_109', 'city_111', 'city_114', 'city_115',
       'city_116', 'city_117', 'city_118', 'city_120', 'city_121',
       'city_123', 'city_126', 'city_127', 'city_128', 'city_129',
       'city_131', 'city_133', 'city_134', 'city_136', 'city_138',
       'city_139', 'city_140', 'city_141', 'city_142', 'city_143',
       'city_144', 'city_145', 'city_146', 'city_149', 'city_150',
       'city_152', 'city_155', 'city_157', 'city_158', 'city_159',
       'city_160', 'city_162', 'city_165', 'city_166', 'city_167',
       'city_171', 'city_173', 'city_175', 'city_176', 'city_179',
       'city_180']
city = st.sidebar.selectbox(
    label = 'Select the city of the employee.',
    options = unique_cities,
    format_func = format_city
)
# Gender
# unique_genders = sorted([x for x in train['gender'].unique().tolist() if str(x)!='nan'])
unique_genders = ['Female', 'Male', 'Other']
gender = st.sidebar.selectbox(
    label = 'Select the gender of the employee.',
    options = unique_genders
)
# Relevent experience
# unique_relevent_experiences = sorted([x for x in train['relevent_experience'].unique().tolist() if str(x)!='nan'])
unique_relevent_experiences = ['Has relevent experience', 'No relevent experience']
relevent_experience = st.sidebar.selectbox(
    label = 'Select whether the employee has relevant experience.',
    options = unique_relevent_experiences
)
# Enrolled university
# unique_enrolled_universities = sorted([x for x in train['enrolled_university'].unique().tolist() if str(x)!='nan'])
unique_enrolled_universities = ['Full time course', 'Part time course', 'no_enrollment']
enrolled_university = st.sidebar.selectbox(
    label = 'Select whether the employee enrolled in a university.',
    options = unique_enrolled_universities,
    format_func = format_cap
)
# Education level
# unique_education_levels = sorted([x for x in train['education_level'].unique().tolist() if str(x)!='nan'])
unique_education_levels = ['Graduate', 'High School', 'Masters', 'Phd', 'Primary School']
education_level = st.sidebar.selectbox(
    label = 'Select the education level of the employee.',
    options = unique_education_levels
)
# Major discipline
# unique_major_disciplines = sorted([x for x in train['major_discipline'].unique().tolist() if str(x)!='nan'])
unique_major_disciplines = ['Arts', 'Business Degree', 'Humanities', 'No Major', 'Other',
       'STEM']
major_discipline = st.sidebar.selectbox(
    label = 'Select the major discipline of the employee.',
    options = unique_major_disciplines
)
# Experience
experience = st.sidebar.number_input(
    label = "Enter the employee's number of years of experience.",
    min_value = 0.,
    max_value = 80.,
    step = 0.1
)
# Company size
company_size = st.sidebar.number_input(
    label = "Enter the employer's current company size.",
    min_value = 1.
)
# Company type
# unique_company_types = sorted([x for x in train['company_type'].unique().tolist() if str(x)!='nan'])
unique_company_types = ['Early Stage Startup', 'Funded Startup', 'NGO', 'Other',
       'Public Sector', 'Pvt Ltd']
company_type = st.sidebar.selectbox(
    label = "Select the employer's company type.",
    options = unique_company_types
)
# Last new job
# unique_lastNewJobs = sorted([x for x in train['last_new_job'].unique().tolist() if str(x)!='nan'])
unique_lastNewJobs = ['1', '2', '3', '4', '>4', 'never']
last_new_job = st.sidebar.selectbox(
    label = "Select the difference in years between the employee's previous job and their current job.",
    options = unique_lastNewJobs
)
# Training hours
training_hours = st.sidebar.number_input(
    label = "Enter the employee's training hours.",
    min_value = 0.,
    step = 1.
)
# City development index
city_development_index = st.sidebar.number_input(
    label = "Enter the city development index.",
    min_value = 0.,
    max_value = 1.,
    step = 0.001
)
data = {'city':city,
        'city_development_index':city_development_index,
        'gender':gender,
       'relevent_experience':relevent_experience,
        'enrolled_university':enrolled_university,
        'education_level':education_level,
       'major_discipline':major_discipline,
        'experience':experience,
        'company_size':company_size,
        'company_type':company_type,
       'last_new_job':last_new_job,
        'training_hours':training_hours}
# Create dataframe
st_features = pd.DataFrame(data, index=[0])
# Format some columns
st_features['experience'] = st_features['experience'].apply(lambda x: experience_st_func(x))
st_features['company_size'] = st_features['company_size'].apply(lambda x: company_size_st_func(x))

##### Run preprocessing pipeline on this one sample of features
st_features_categorical = st_features.drop(['city_development_index','training_hours'],axis=1)
# One-hot encode
st_features_enc_array = encoder.transform(st_features_categorical).toarray()
# Recreate X_train dataframe
st_features_enc = pd.DataFrame(
    st_features_enc_array, columns = colnames
)
st_features_enc['city_development_index'] = st_features['city_development_index']
# Scale training_hours
st_features_scaled_hours_smote = scaler_smote.transform(np.array(st_features['training_hours']).reshape(-1, 1))
st_features_enc_smote = st_features_enc.copy()
st_features_enc_smote['scaled_train_hours'] = list(st_features_scaled_hours_smote.flat)
# PCA
PCA_st_features_enc_smote = pca_smote.transform(st_features_enc_smote.values)
# Predict
st_pred = best_clf.predict(PCA_st_features_enc_smote)
# Convert to text string
st_pred_str = ['**not** looking for a job change' if st_pred[0]==0 else 'looking for a job change' for x in st_pred][0]

##### Display result
st.subheader('**Prediction:**')
st.write('The employee is {}.'.format(st_pred_str))