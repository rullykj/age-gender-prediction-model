import os
import numpy as np
import pandas as pd

import cv2
import face_recognition

imdb_csv = 'meta_imdb.csv'
aligned_imdb_csv = 'aligned_imdb.csv'
output_folder = 'imdb_align'

cols = ['age', 'age_category', 'gender', 'gender_id', 'path']

try: 
    os.mkdir(output_folder) 
except OSError as error: 
    print(error)  

imdb_pd = pd.read_csv(imdb_csv, sep=';')

image_paths = imdb_pd['path']
ages = imdb_pd['age']
age_categories = imdb_pd['age_category']
genders = imdb_pd['gender']
gender_ids = imdb_pd['gender_id']

imdb_path = []
imdb_age = []
imdb_age_category = []
imdb_genders = []
imdb_genders_id = []

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
            imdb_age.append(age)
            imdb_age_category.append(age_category)
            imdb_genders.append(gender)
            imdb_genders_id.append(gender_id)
            imdb_path.append(os.path.join(output_folder,str(path).split(os.sep)[2]))
        
final_imdb = np.vstack((imdb_age, imdb_age_category, imdb_genders, imdb_genders_id, imdb_path)).T

final_imdb_df = pd.DataFrame(final_imdb)

final_imdb_df.columns = cols

final_imdb_df.to_csv(aligned_imdb_csv, index=False, sep=";")