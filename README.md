# Hand_Sign_language_recognition
 This repository is complete package to hand sign recognition made in python using libraries such as openCV and tensorflow etc..
 
# FILES DESCRIPTION :

# dataset_creator.py :
    This file containing the code that works as an dataset creator which means it captures the images using your cam up to the number of times you specify.
    The file format is :
    folder..Dataset / symbol_name(specified by you) / images(captured)....upto number of times specified.

# model_creator.py :
    This file contains the code that uses the Dataset created by "dataset_creator.py" to train and create a model named "model.h5" that will be used in the recognition part of the code.
    file created : model.h5

# Recognition.py :
    This file contains the code that uses the trained model "model.h5" created by "model_creator.py" to recognise the symbol specified in dataset and model.


# NOTE : 
    ALWAYS KEEP THESE FILES (the code files & the files created ( Dataset folder / model.h5 ) ) INTO A SINGLE FOLDER.
IF NOT POSSIBLE PLEASE DO ASSIGN THE PATH SPECIFIED TO MAKE THE CODE WORK.


# THANK YOU 😊
