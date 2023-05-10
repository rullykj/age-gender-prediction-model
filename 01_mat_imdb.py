import os
import platform
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['age', 'age_category', 'gender', 'gender_id', 'path', 'face_score1', 'face_score2']

imdb_mat = 'imdb_crop/imdb.mat'

imdb_data = loadmat(imdb_mat)

del imdb_mat

imdb = imdb_data['imdb']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

imdb_path = []

def change_delimiter(old_path):
    if platform.system() == "Windows":
        new_path = old_path.replace("/", os.sep)
        return new_path
    return old_path

for path in imdb_full_path:
    imdb_path.append(os.path.join('imdb_crop', change_delimiter(path[0])))

imdb_genders = []
imdb_genders_id = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
        imdb_genders_id.append(0)
    else:
        imdb_genders.append('female')
        imdb_genders_id.append(1)

imdb_dob = []

for file in imdb_path:
    temp = file.split('_')[3]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]

    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    
    imdb_dob.append('-'.join(temp))


imdb_age = []
imdb_age_category = []

for i in range(len(imdb_dob)):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)
    if diff>=1 and diff<=2:
        imdb_age_category.append(0)
    elif diff>=3 and diff<=9:
        imdb_age_category.append(1)
    elif diff>=10 and diff<=20:
        imdb_age_category.append(2)
    elif diff>=21 and diff<=27:
        imdb_age_category.append(3)
    elif diff>=28 and diff<=45:
        imdb_age_category.append(4)
    elif diff>=46 and diff<=65:
        imdb_age_category.append(5)
    elif diff>=66 and diff<=116:
        imdb_age_category.append(6)
    else:
        imdb_age_category.append(7)

final_imdb = np.vstack((imdb_age, imdb_age_category, imdb_genders, imdb_genders_id, imdb_path, imdb_face_score1, imdb_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)

final_imdb_df.columns = cols

meta = final_imdb_df

meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']

meta = meta.drop(['face_score1', 'face_score2'], axis=1)

meta = meta.sample(frac=1)

meta.to_csv('meta_imdb.csv', index=False, sep=";")

