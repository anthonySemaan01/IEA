# Character/Digit Recognizer

![image](https://user-images.githubusercontent.com/92089618/198839952-7636aeda-f613-4133-b25d-f0815a59cf9f.png)

CAM is a character/ digit recognizer web application and its purpose is to help kids learn writing letters and digits and by translating their characters into Arabic.
Hence, CAM is a modern day translator that aims to broaden the learning of the arabic language for children around the world!

Character Classifier is an ensemble learning model used to classify Characters and digits


## Installation

Clone the repo and install all the requirements to get the dependencies needed.
We advise to create a virtual environment with python 3.9 as interpreter.

[csv_ensemble1.zip](https://github.com/anthonySemaan01/IEA/files/9920005/csv_ensemble1.zip)

[csv_ensemble2.zip](https://github.com/anthonySemaan01/IEA/files/9919993/csv_ensemble2.zip)

Download and extract these 2 zip folders. CSV files in ensemble1 put them in datasets/training/vector. CSV files in ensemble2 put them in datasets/training/vector2



```bash
pip install -r requirements.txt
```

## File Structure
This API follows a Domain Driven Structure with some changes:
```bash
.
├── api
│   └── controllers
├── application
│   ├── feature_extraction
│   ├── image_preprocessing
│   └── inference
├── datasets
│   ├── cropped_images
│   ├── eroded_dilated
│   ├── gray_scaled_images
│   ├── images
│   ├── processed
│   ├── resized_images
│   ├── testing
│   └── training
│       ├── vector
│       └── vector2      
├── domain
│   ├── contracts
│   ├── exceptions
│   └── models
├── model
├── persistence
│   └── repositories
├── shared
│   ├── data_handler
│   └── helper
├── containers.py
├── main.py
└── README.md
```


## Usage

In the root directory of the project, type the following command in the terminal

```bash
uvicorn main:app --reload
```

this will start the API at port 8000. For easier visualization, surf to http://localhost:8000/docs
This will give a simple representation of the available endpoints.

### Endpoints

**Health**: Check if the API is working correctly

**image_preprocessing**: Post request which takes an image and preprocess it based on some predefined parameters
the output series of images are in the ./datasets directory

**preprocessing_training_dataset**: Get request which takes the root directory of your training dataset. The
preprocessed dataset will be saved in the root directory

**retrain**: Post request which takes an image and a label. The following pair will be added to different csv files
accounting for the submitted label. In case of error, no come-back is possible.

**preprocessing_inference**: Post request which takes an image and outputs the classification result.
The preprocessing of the image is done in the background.

## Acknowledgments

- [Anthony Semaan](https://github.com/anthonySemaan01), Lebanese American University, Byblos, Lebanon
- [Michael Al Assaad](https://github.com/michaelalassaad), Lebanese American University, Byblos, Lebanon
- [Charbel el Chidiac](https://github.com/charbelc15), Lebanese American University, Byblos, Lebanon


