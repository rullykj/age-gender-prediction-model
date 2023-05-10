# age-gender-prediction-model

This is a repository for version control of the Model training and testing of the for Team 9 - MSC group project.

This project is a part of my coursework at University of Liverpool.

The rights of this project are reserved by :
- Aravind Reddy Annapureddy 
- Atri Gulati 
- Chiamaka Jibuaku
- Rully Kusumajaya 
- Dheraz Rohinton Luth 
- Odianosen Masade 
- Herald Olakkengil 
- Joy Onuoha

We use UTKFace for training:
- https://susanqq.github.io/UTKFace/

and imdb-wiki for testing the model:
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

## Training
Using google colab

## Testing
- Parsing from MatLab mat to csv
	- 01_mat_imdb.py
	- 01_mat_wiki.py
- Align and crop using dlib
	- 02_align_crop_imdb.py
	- 02_align_crop_wiki.py
- Predict using model
	- 03_predict_align_crop_imdb.py
	- 03_predict_align_crop_wiki.py 
 
