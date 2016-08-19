Ultrasound Nerve Segmentation
=============================

Capstone Project for Udacity MLND


### Environment Requirements

The following needs to be installed:

- Python 2.7, or >= 3.4
- numpy >= 1.11
- Tensorflow >= 0.9
- OpenCV >= 2.4
- tqdm >= 4.8

NOTE: tqdm can be installed using ```pip install tqdm```


### Downloading the Dataset

The dataset can be downloaded from Kaggle competetion "Ultrasound Nerve Segmation":

https://www.kaggle.com/c/ultrasound-nerve-segmentation/data

Download the ***train*** set and unzip the contents. Put the images (.tif files) in ```<root_project_dir>/data/train/```.


### Preprocessing
After downloading the dataset, run the preprocess routine by issuing the command:

```bash
python preprocess.py
```

After running the script, the following files will be generated:
```
data/train_xs96/        - Directory containing the filtered and resized images
data/train_set.npz      - Train set in numpy readable format
data/validation_set.npz - Validation set in numpy readable format
data/train_stats.pkl    - Pickle file that contains basic statistics about the train set
```



### Training the Model

To train the model, execute the command:

```bash
python train.py
```

Training takes approximately 17 minutes per epoch on Amazon AWS g2.2xlarge instance with GPU support. A pre-trained model, that was ran for 10 hours with 35 epochs, can be downloaded at:

https://www.dropbox.com/s/xoib4r58hnbwkan/model-35.zip?dl=0

Extract the files and place the ***'model.ckpt-35'*** and ***'model.ckpt-35.meta'*** files in ```output/checpoints/``` directory.



### Running Inference

To run inference, place the target images (.tif files) in a directory. If the images have accompanying ground truth labels, the labels must have the same name as the target and end with ```<name>_mask.tif```.

Execute the command:

```bash
python inference.py -s <image_dir>
```
The generated output will be saved in ```<image_dir>_output```

For more details on accepted arguments, see ```inference.py``` file.


A sample set of images can be found at ```sample_data/``` directory. To see a smaple output, execute the command:

```bash
python inference.py -s sample_data/
```

The generated output can be found at ```sample_data_output/```.
