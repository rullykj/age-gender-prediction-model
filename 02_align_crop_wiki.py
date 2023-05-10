import os
import numpy as np
import pandas as pd

import cv2
import face_recognition

wiki_csv = 'meta_wiki.csv'
aligned_wiki_csv = 'aligned_wiki.csv'
output_folder = 'wiki_align'

cols = ['age', 'age_category', 'gender', 'gender_id', 'path']

try: 
    os.mkdir(output_folder) 
except OSError as error: 
    print(error) 
    
wiki_pd = pd.read_csv(wiki_csv, sep=';')

image_paths = wiki_pd['path']
ages = wiki_pd['age']
age_categories = wiki_pd['age_category']
genders = wiki_pd['gender']
gender_ids = wiki_pd['gender_id']

wiki_path = []
wiki_age = []
wiki_age_category = []
wiki_genders = []
wiki_genders_id = []

def align_crop(image_path):
    result = False

    cv_image = cv2.imread(image_path)
    image_h, image_w = cv_image.shape[0], cv_image.shape[1]
#     margin = 0.01
    margin = 0

    face_locations = face_recognition.face_locations(cv_image, model='hog')

    if len(face_locations) == 1:
#         face_batch = np.empty((len(face_locations), 200, 200, 3))

        # add face images into batch
        for i,rect in enumerate(face_locations):
            # crop with a margin
            top, bottom, left, right = rect[0], rect[2], rect[3], rect[1]
            top = max(int(top - image_h * margin), 0)
            left = max(int(left - image_w * margin), 0)
            bottom = min(int(bottom + image_h * margin), image_h - 1)
            right = min(int(right + image_w * margin), image_w - 1)

            face_img = cv_image[top:bottom, left:right, :]
            face_img = cv2.resize(face_img, (200, 200))
#             face_batch[i, :, :, :] = face_img
            path = os.path.join(output_folder,str(image_path).split(os.sep)[2])
            cv2.imwrite(path, face_img)
            result = True
    
    return result

for age, age_category, gender, gender_id, path in zip(ages, age_categories, genders, gender_ids, image_paths):
    if age_category<7:
        if align_crop(path):
            wiki_age.append(age)
            wiki_age_category.append(age_category)
            wiki_genders.append(gender)
            wiki_genders_id.append(gender_id)
            wiki_path.append(os.path.join(output_folder,str(path).split(os.sep)[2]))
        
final_wiki = np.vstack((wiki_age, wiki_age_category, wiki_genders, wiki_genders_id, wiki_path)).T

final_wiki_df = pd.DataFrame(final_wiki)

final_wiki_df.columns = cols

final_wiki_df.to_csv(aligned_wiki_csv, index=False, sep=";")