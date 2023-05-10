import os
import platform
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['age', 'age_category', 'gender', 'gender_id', 'path', 'face_score1', 'face_score2']

wiki_mat = 'wiki_crop/wiki.mat'

wiki_data = loadmat(wiki_mat)

del wiki_mat

wiki = wiki_data['wiki']

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

wiki_path = []

def change_delimiter(old_path):
    if platform.system() == "Windows":
        new_path = old_path.replace("/", os.sep)
        return new_path
    return old_path

for path in wiki_full_path:
    wiki_path.append(os.path.join('wiki_crop', change_delimiter(path[0])))

wiki_genders = []
wiki_genders_id = []

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
        wiki_genders_id.append(0)
    else:
        wiki_genders.append('female')
        wiki_genders_id.append(1)

wiki_dob = []

for file in wiki_path:
    wiki_dob.append(file.split('_')[2])


wiki_age = []
wiki_age_category = []

for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    wiki_age.append(diff)
    if diff>=1 and diff<=2:
        wiki_age_category.append(0)
    elif diff>=3 and diff<=9:
        wiki_age_category.append(1)
    elif diff>=10 and diff<=20:
        wiki_age_category.append(2)
    elif diff>=21 and diff<=27:
        wiki_age_category.append(3)
    elif diff>=28 and diff<=45:
        wiki_age_category.append(4)
    elif diff>=46 and diff<=65:
        wiki_age_category.append(5)
    elif diff>=66 and diff<=116:
        wiki_age_category.append(6)
    else:
        wiki_age_category.append(7)

final_wiki = np.vstack((wiki_age, wiki_age_category, wiki_genders, wiki_genders_id, wiki_path, wiki_face_score1, wiki_face_score2)).T

final_wiki_df = pd.DataFrame(final_wiki)

final_wiki_df.columns = cols

meta = final_wiki_df

meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']

meta = meta.drop(['face_score1', 'face_score2'], axis=1)

meta = meta.sample(frac=1)

meta.to_csv('meta_wiki.csv', index=False, sep=";")

