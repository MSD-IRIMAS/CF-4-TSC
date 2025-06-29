> ⚠️ **Alert:** If you are using this code with **Keras v3**, make sure you are using **Keras ≥ 3.6.0**.  
> Earlier versions of Keras v3 do not honor `trainable=False`, which will result in **training hand-crafted filters** in **CO-FCN** **H-FCN** and **H-Inception** unexpectedly.

### Using [aeon-toolkit](https://github.com/aeon-toolkit/aeon) to Train a H-InceptionTime on your DATA

When using aeon, simply load your data and create an instance of a InceptionTime classifier while specifyin the usage of hand-crafted filters and train it <br>

```
from aeon.datasets import load_classification
from aeon.classification.deep_learning import InceptionTimeClassifier

xtrain, ytrain, _ = load_classification(name="Coffee", split="train")
xtest, ytest, _ = load_classification(name="Coffee", split="test")

# using H-InceptionTime
clf = InceptionTimeClassifier(n_classifiers=5, use_custom_filters=True)
clf.fit(xtrain, ytrain)
print(clf.score(xtest, ytest)
```

# Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters

This is the code of our paper "[Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters](https://doi.org/10.1109/BigData55660.2022.10020496)" accepted as a regular paper at [2022 IEEE Internation Conference on Big Data](https://bigdataieee.org/BigData2022/).<br>
This work was done by [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Maxime Devanne](https://maxime-devanne.com/), [Jonathan Weber](https://www.jonathan-weber.eu/) and [Germain Forestier](https://germain-forestier.info/).<br><br>

A web page of our paper can be found [here](http://maxime-devanne.com/pages/HCCF-4-tsc/).

## Summary of proposed custom filters

<p align="center" width="100%">
<img src="images/summary.png" alt="summary"/>
</p>

## Architectures:

### Architecture 1: Custom Only Fully Convolutional Network (CO-FCN)

<p align="center">
  <img src="images/CO-FCN.png" />
</p>


### Architecture 2: Hybrid Fully Convolutional Network (H-FCN)

<p align="center">
  <img src="images/H-FCN.png" />
</p>

### Architecture 3: Hybrid Inception (H-Inception) and Hybrid InceptionTime (H-InceptionTime)<br>
Note: H-InceptionTime is an ensemble of five H-Inception models, such as in [H. Ismail Fawaz et al.](https://github.com/hfawaz/InceptionTime)

<p align="center">
  <img src="images/H-Inception.png" />
</p>


## Usage of code

### DOCKER IMAGE

Now you can use docker image to run the code instead of using pip or conda environments.

To build the image run the following <br>

```docker build -t IMAGE_NAME .```<br>

To build and run the container with mounted directories for the data and the code, run the following: <br>

```docker run --gpus all -it --name CONTAINER_NAME -v "$(pwd):/cf4tsc-code" -v "/path/to/ucr/on/your/pc:/ucr_archive" IMAGE_NAME bash```<br>

This will open a shell where you can execute the code.

### CODE EXECUTION

In order to run an experiment on a dataset of the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) with a specific classifier of three proposed above, simply run the ```main.py``` file. <br>
This file takes as arguments the following:<br>
```
--dataset : The dataset to be used, by default "Coffee" is used<br>
--classifier : The classifier to be used, choices = ['CO-FCN', 'H-FCN', 'H-Inception'], by default 'H-Inception' is used
--runs : The number of experiments to be done, by default five are done
--output-directory : The output directory, by default the output is saved into 'results/'
```

### Adaptation of code

The change that should be done is the directory in which the datasets are stored.<br>
The variable to be changed is [in this line](https://github.com/MSD-IRIMAS/CF-4-TSC/blob/4acebb612786c43d4f72d4c3de3b0d97667c0625/utils/utils.py#L12) ```folder_path```.

## Results

### CO-FCN -- see [results csv](results/CO-FCN/results_UCR_128.csv) on the 128 datasets of the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)

<p align="center">
  <img style="height: auto; width: 300px" src="images/results_cofcn.png" />
</p>


### H-FCN -- see [results csv](results/H-FCN/results_UCR_128.csv) on the 128 datasets of the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)


<p align="center">
  <img style="height: auto; width: 600px" src="images/results_hfcn.png" />
</p>

### H-Inception -- see [results csv](results/H-Inception/results_UCR_128.csv) on the 128 datasets of the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)


<p align="center">
  <img style="height: auto; width: 600px" src="images/results_hinception.png" />
</p>

## Critical Difference Diagram, using the code of [H. Ismail Fawaz et al.](https://github.com/hfawaz/cd-diagram)

<p align="center">
  <img style="height: auto; width: 600px" src="images/cd-diagram.png" />
</p>


## Matrix 1v1 Comparison

<p align="center">
  <img style="height: auto; width: 900px" src="images/heatmap.png" />
</p>


## Reference

If you use this code, please cite our paper:<br>

```
@inproceedings{ismail-fawaz2022hccf,
  author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber, Jonathan and Forestier, Germain},
  title = {Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters},
  booktitle = {2022 IEEE International Conference on Big Data (IEEE BigData 2022)},
  city = {Osaka},
  country = {Japan},
  pages = {972-981},
  url = {doi.org/10.1109/BigData55660.2022.10020496},
  year = {2022},
  organization = {IEEE}
}
```

## Acknowledgments

This work was supported by the ANR DELEGATION project (grant ANR-21-CE23-0014) of the French Agence Nationale de la Recherche. The authors would like to acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific support and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d’Avenir) and the CPER Alsacalcul/Big Data. The authors would also like to thank the creators and providers of the UCR Archive.
