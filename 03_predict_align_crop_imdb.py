import os
import numpy as np
import pandas as pd

import cv2
import face_recognition
import tensorflow as tf
# import onnxruntime as rt

import time

aligned_imdb_csv = 'aligned_imdb.csv'
final_imdb_csv = 'final_imdb.csv'
# https://drive.google.com/file/d/1oad8Bc_yhaoS2TsHBm-tX7NHNg6RydZ8/view
model_path = 'face_weights.05-val_loss-0.90-val_age_loss-0.74-val_gender_loss-0.16.utk.h5'
# https://drive.google.com/file/d/1W8YPeTVv6ISmKIGtowrSbvxPYvwDZznE/view
# onnx_model_path = 'age_gender-resnet50-final.onnx'

MyModel = tf.keras.models.load_model(model_path)
# MyOnnxModelSession = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

cols = ['age', 'age_category', 'gender', 'gender_id', 'path', 'age_predict', 'gender_predict', 'age_result', 'gender_result']

def preprocess_input_resnet50(x):
    x_temp = np.copy(x)
    
    # mean subtraction
    # already BGR in opencv
    #x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 91
    x_temp[..., 1] -= 103
    x_temp[..., 2] -= 131
    
    return x_temp

def predict(image_path):
    # gender_labels = ['Male', 'Female']
    # age_labels = ['1-2','3-9','10-20','21-27','28-45','46-65','66-116']

    cv_image = cv2.imread(image_path)
    image_h, image_w = cv_image.shape[0], cv_image.shape[1]
#     margin = 0.01
    margin = 0

    face_locations = face_recognition.face_locations(cv_image, model='hog')

    if len(face_locations) > 0:
        face_batch = np.empty((len(face_locations), 200, 200, 3))

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
            face_batch[i, :, :, :] = face_img

#         face_batch = tf.keras.applications.resnet50.preprocess_input(face_batch)
        face_batch = preprocess_input_resnet50(face_batch)
        
        preds = MyModel.predict(face_batch)

        return preds

# def predict_onnx(image_path):
#     cv_image = cv2.imread(image_path)
#     image_h, image_w = cv_image.shape[0], cv_image.shape[1]

#     margin = 0

#     face_locations = face_recognition.face_locations(cv_image, model='hog')

#     if len(face_locations) > 0:
#         face_batch = np.empty((len(face_locations), 200, 200, 3))

#         # add face images into batch
#         for i,rect in enumerate(face_locations):
#             # crop with a margin
#             top, bottom, left, right = rect[0], rect[2], rect[3], rect[1]
#             top = max(int(top - image_h * margin), 0)
#             left = max(int(left - image_w * margin), 0)
#             bottom = min(int(bottom + image_h * margin), image_h - 1)
#             right = min(int(right + image_w * margin), image_w - 1)

#             face_img = cv_image[top:bottom, left:right, :]
#             face_img = cv2.resize(face_img, (200, 200))
#             face_batch[i, :, :, :] = face_img

#         face_batch = preprocess_input_resnet50(face_batch)
        
#         onnx_pred = MyOnnxModelSession.run(['predications_age', 'predications_gender'], {"input_5": face_batch.astype(np.float32)})
        
#         return onnx_pred


imdb_pd = pd.read_csv(aligned_imdb_csv, sep=';')

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

imdb_age_predict = []
imdb_gender_predict = []

imdb_age_result = []
imdb_gender_result = []


start_time = round(time.time()*1000)
# count = 0
for age, age_category, gender, gender_id, path in zip(ages, age_categories, genders, gender_ids, image_paths):
    preds = predict(path) #tensorflow
    # preds = predict_onnx(path) #onnx

    # print(preds)
    if(preds is not None):
        preds_ages = preds[0]
        preds_genders = preds[1]

        age_index = np.argmax(preds_ages)
        gender_index = np.argmax(preds_genders)

        imdb_age_predict.append(age_index)
        imdb_gender_predict.append(gender_index)

        imdb_age.append(age)
        imdb_age_category.append(age_category)
        imdb_genders.append(gender)
        imdb_genders_id.append(gender_id)
        imdb_path.append(path)
        if age_category == age_index:
            imdb_age_result.append(True)
        else:
            imdb_age_result.append(False)
        if gender_id == gender_index:
            imdb_gender_result.append(True)
        else:
            imdb_gender_result.append(False)
    #     count+=1
    # if count==100:
    #     break

end_time = round(time.time()*1000)
print("Milliseconds :",str(end_time-start_time))
        
final_imdb = np.vstack((imdb_age, imdb_age_category, imdb_genders, imdb_genders_id, imdb_path, imdb_age_predict, imdb_gender_predict, imdb_age_result, imdb_gender_result)).T

final_imdb_df = pd.DataFrame(final_imdb)

final_imdb_df.columns = cols

final_imdb_df.to_csv(final_imdb_csv, index=False, sep=";")