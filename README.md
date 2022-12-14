# Character/Digit Recognizer

![image](https://user-images.githubusercontent.com/92089618/198839952-7636aeda-f613-4133-b25d-f0815a59cf9f.png)

CAM is a character/ digit recognizer web application and its purpose is to help kids learn writing letters and digits
and by translating their characters into Spanish.
Hence, CAM is a modern day translator that aims to broaden the learning of the Spanish language for children around the
world!

As part of our initial analysis, we designed the character/digits classifier as an ensemble learning model based only on
non-parametric learners.
Ensemble Model 2 reached an 84% accuracy level tested on 500 labels.

In order to diversify our analysis, we tested different parametric learners architectures, starting with single layer
perceptron to Multi Layer perceptron. To simplify the integration of the models with the API, we created an endpoint for
a weighted sum ensemble learning model based on 3 different MLP architectures.

## Installation

Clone the repo and install all the requirements to get the dependencies needed.
We advise to create a virtual environment with python 3.9 as interpreter.

[csv_ensemble1.zip](https://github.com/anthonySemaan01/IEA/files/9920005/csv_ensemble1.zip)

[csv_ensemble2.zip](https://github.com/anthonySemaan01/IEA/files/9919993/csv_ensemble2.zip)

These weights are used for both models 1 and 2. For the rest of the models, downloading the weights in ./weights is
sufficient.

Download and extract these 2 zip folders. CSV files in ensemble1 put them in datasets/training/vector. CSV files in
ensemble2 put them in datasets/training/vector2

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
│  
├── weights 
├── notebooks
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

## Acknowledgments

- [Anthony Semaan](https://github.com/anthonySemaan01), Lebanese American University, Byblos, Lebanon
- [Michael Al Assaad](https://github.com/michaelalassaad), Lebanese American University, Byblos, Lebanon
- [Charbel el Chidiac](https://github.com/charbelc15), Lebanese American University, Byblos, Lebanon


