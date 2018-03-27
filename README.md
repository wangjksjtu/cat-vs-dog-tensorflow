# cat-vs-dog-tensorflow
An exploration of the trade-off between image quality and accuracy.

## Prepare the data
The dataset is downloaded from Kaggle ([https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)).
After getting the data, move the train.zip file to `data/` and decompress the file. The structure of `data/` should be like as follows:

```
├── __init__.py
├── prepare.py
├── quantize.py
├── test.zip (not used)
├── train.zip
└── train
    ├── cat.m.jpg [12500 images]
    └── dog.n.jpg [12500 images]
```

Then the images of different qualities and correspondingly prepared h5 files (e.g. `quality_0\data.h5`) could be generated using `quantize.py` and `prepare.py`, respecitively.

__Note__: For this dataset, h5 file is not a very good choice because generated `data.h5` is too large (`25GB`) and it also takes a lot of time to read the data into the memory (`> 4GB`). I will search for more efficient ways of utilizing data.

## Original Model
From this time, the model is a simple multi-layer CNNs in order to shrink the parameters easily. The architecure of model is given in `model.py`. Or you can obatin the [architecure](https://github.com/wangjksjtu/cat-vs-dog-tensorflow/blob/master/parameters/models/model_22170834_original.pdf) and [parameters](https://github.com/wangjksjtu/cat-vs-dog-tensorflow/blob/master/parameters/models/model_22170834_original.txt) in `parameters/models/`.

## Train
To train the model on images with quality scale of 5, just type the following command in your bash.

    python train.py --quality 5
    python train_paras.py --setting 2 (for parameter minimizing)
    
Besides image quality, other hyper-parameters such as learning rate and training epochs rate can also be specified. You can get more information by adding `-h` while runing the script: 

    python train.py -h
    python train_paras.py -h (for parameter minimizing)

Having finished trainiing, you could collect the result using `logs/collect.py` script like:

    cd logs/
    python collect.py --quality 15 --setting 1

It is worthy to notice that the relationship between settings and models are list in `model_paras.py`.

## Plot
After having trained the models, you obatin the results using `logs/collect.py` script. Then the collected results should be set in `plot.py` for plotting.

    python plot.py

## Results
### Architecure of 10 models
The detailed information about different models for parameter minimizing is given in `parameters/models/`.
To reproduce the results (architecure figure & parameters txt), run the `parameters/model.py` scirpt.

    cd parameters
    python model.py --all True
    
### Precision VS Quality
<p align="center">
  <img src="https://github.com/wangjksjtu/cat-vs-dog-tensorflow/blob/master/results/precision_quality.png">
</p>

### Minimizing the parameters 
<p align="center">
  <img width=725 src="https://github.com/wangjksjtu/cat-vs-dog-tensorflow/blob/master/results/parameters.png">
</p>
