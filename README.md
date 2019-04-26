# Masters Project
Masters Project on Transfer Learning in CNNs with Medical Images

## Project Thesis
[Click here to read my Thesis](Thesis.pdf)
[Click here to view my defense slides](Project_Defense_Slides.pdf)

## Datasets
Kaggle Medical Image Datasets
1. [HAM10000 Skin Cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
2. [Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Files
**Please change all path directories when using the files**
1. [augment_data.py](augment_data.py) is use to testing augmenting data when training.
2. [image_preprocessing_test.py] (image_preprocessing_test.py) was used to test if equalization the images benefits the learning.
3. [model_utils.py](model_utils.py) contains models utils methods.
4. [my_model.py](my_model.py) was used as a test model.
5. [plot.py](plot.py) used to create line graph of training with ImageNet weights compared to random weights.
6. [pretrained_cnn.py](pretrained_cnn.py) uses pretained ImageNet weights to create a model.
7. [random_forest.py](random_forest.py) used to create baseline models: Decision Tree and Random Forest.
8. [split_data.py](split_data.py) used to split the HAM dataset into training/validation/test sets.
9. [test_output.py](test_output.py) used to load model and test the accuracy from the test set.
10. [xray_cnn.py](xray_cnn.py) used to create model for pretained and random weights.
11. [xray_cnn_w_HAM.py](xray_cnn_w_HAM.py) used to load HAM weights and create a model.
12. [xray_data_preprocess.py](xray_data_preprocess.py) used to preprocess the data and split into training/validation/test sets.
13. [xray_random_decision.py](xray_random_decision.py) used to create baseline models: Decision Tree and Random Forest.
