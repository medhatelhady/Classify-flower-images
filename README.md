
# Deep Learning
## Classify flowers using pytorch




### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [matplotlib](http://matplotlib.org/)
- [json](https://www.json.org/)

You will also need to have software installed to run and execute an iPython Notebook

We recommend students install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Objectives
use transfer learning techniques to update pre-trained model (resnet) to predict flowers type. 




### Run

to run the program, if you would like train the model, then run `train.py` and specifiy 
* image directory
* model arch
* data directory
* laerning rate
* number of hidden units
* epochs
* use gpu or cpu

you have multiple options:
* Set directory to save checkpoints: 
`python train.py data_dir --save_dir save_directory`
* Choose architecture: 
`python train.py data_dir --arch "vgg13"`
* Set hyperparameters: 
`python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: 
`python train.py data_dir --gpu`


in order to predict result, then run `predict.py` and specify 
Options:
* Return top KK most likely classes: 
`python predict.py input checkpoint --top_k 3`
* Use a mapping of categories to real names: 
`python predict.py input checkpoint --category_names cat_to_name.json`
* Use GPU for inference: 
`python predict.py input checkpoint --gpu`


