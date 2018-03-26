# cat-vs-dog-tensorflow
An exploration of the trade-off between image quality and accuracy.

## Prepare the data
The dataset is downloaded from Kaggle ([https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)).
After getting the data, move the train.zip file to `data/` and decompress the file. The structure of `data/` should be like as follows:

```
├── __init__.py
├── prepare.py
├── quantize.py
├── test.zip
├── test [12500 images]
|   ├── cat.m.jpg [12500 images]
|   └── dog.n.jpg [12500 images]
├── train.zip
└── train
    ├── cat.m.jpg [12500 images]
    └── dog.n.jpg [12500 images]
```

Then the images of different qualities and correspondingly prepared h5 files (e.g. `quality_0\data.h5`) could be generated using `quantize.py` and `prepare.py`, respecitively.

__Note__: For this dataset, h5 file is not a very good choice because generated `data.h5` is too large (`25GB`) and it also takes a lot of time to read the data into the memory (`> 4GB`). I will search for more efficient ways of utilizing data.

## Model
From this time, the model is a simple multi-layer CNNs in order to shrink the parameters easily. The architecure of model is given in `model.py`.

## Train
To train the model on images with quality scale of 5, just type the following command in your bash.

    python train.py --quality 5
    
Besides image quality, other hyper-parameters such as learning rate and training epochs rate can also be specified. You can get more information by adding `-h` while runing the script: 

    python train.py -h

## Plot
After having trained the models, you obatin the results using `logs/collect.py` script. Then the collected results should be set in `plot.py` for plotting.

    python plot.py
