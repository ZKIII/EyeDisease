# SYDE 522 Project



## Basic Requirements

* OS: Windows

* [Python Version: 3.12.2(64-bit)](https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe)

* Packages

  > All required packages are listed in the `requirements.txt` file.

  * Install packages with command line: `pip install -r requirements.txt`





## Python Scripts Introduction

* `DownloadData.py`
  * Download eye diseases images data and trained models from google drive. Create necessary folders. <span style="color: red;">There is a limitation of download file by gdown. If download fail, you may need to download and unzip [`Data.zip`](https://drive.google.com/file/d/1HTOOXIrf4iFd88u2gdCI7jPovKhSGRYx/view?usp=drive_link), [`Saved_Models.zip`](https://drive.google.com/file/d/1862QUN49PRLHCAuLBFew8U9YBJVN8sqw/view?usp=drive_link) and ['Losses_Acc.zip'](https://drive.google.com/file/d/163qWhbsJfH-VzBk4CEhzCaC93RwWtc_Z/view?usp=drive_link) manually. Make sure the path is correct(`Data/train/*.png`, `Data/dataset.csv`, `Data/all_data.pkl`, `Saved_Models/*.pth`, `Losses_Acc/*.pkl`).</span>

* `DataTransform.py`
  * Class `DataTransform` loads original data sets and splits them into training and test sets. All data sets will be saved into `Data/all_data.pkl`.
  * Class `DataTransformSet` is used to load data sets such as PyTorch data set structures.
* `Models.py`
  * Class `ViT` is implemented by a [pre-trained structure of Vision Transformer](https://huggingface.co/google/vit-base-patch16-224-in21k)
  * Class `ResNet50` is implemented by a pre-trained structure of ResNet50 of `torchvision.models.resnet50`
* `Train.py`
  * The scripts of training model processes by provide suitable batch size, learning rate and epochs.
* `Evaluation.py`
  * The scripts of plot training process losses and accuracy of all models and report all final models' accuracy and f1-scores.





## Folders Introductions

* `Data`:
  * `train/*.jpg`: Including all original image data.
  * `dataset.csv`: Features and labels of our data set.
  * `all_data.pkl`: Saved all images and feature data into a single file for future processing.
* `Losses_Acc`:
  * `*.pkl`: All losses and accuracy during the training processes.
* `Reports`:
  * `Acc_F1.txt`: The final selected models' prediction accuracy and f1-scores.
  * `*.png`: The plot of losses and accuracy during the training processes.
* `Saved_Models`:
  * `*.pth`: Trained models





## Execution

* Introduction of all scripts

  * Run `DownloadData.py` to get all required files. Download eye diseases images data and trained models from google drive. Create necessary folders. <span style="color: red;"><span style="color: red;">There is a limitation of download file by gdown. If download fail, you may need to download and unzip [`Data.zip`](https://drive.google.com/file/d/1HTOOXIrf4iFd88u2gdCI7jPovKhSGRYx/view?usp=drive_link), [`Saved_Models.zip`](https://drive.google.com/file/d/1862QUN49PRLHCAuLBFew8U9YBJVN8sqw/view?usp=drive_link) and ['Losses_Acc.zip'](https://drive.google.com/file/d/163qWhbsJfH-VzBk4CEhzCaC93RwWtc_Z/view?usp=drive_link) manually. Make sure the path is correct(`Data/train/*.png`, `Data/dataset.csv`, `Data/all_data.pkl`, `Saved_Models/*.pth`, `Losses_Acc/*.pkl`).</span>

  * For generating data(e.g., resize the training and test set and shuffle the data set), just run `DataTransform.py`. You may select prefer training and test set size or shuffle data set. The file `all_data.pkl` in `Data` folder will be replaced.

  * For training models, just run `Train.py` or change some other value of learning rate, batch size or epochs. The models in `Saved_Models` will be replaced.

  * For model evaluations, just run `Evaluation.py`. The loss and accuracy file in `Losses_Acc` and plots in `Reports` will be replaced.


* Execution options

  * If you want to use new data to train some new models:

    ```mermaid
    graph LR
    DownloadData.py --> DataTransform.py
    DataTransform.py --> Train.py
    Train.py --> Evaluation.py
    ```

  * If you just want to train new models by pre-processed training and test set:

    ```mermaid
    graph LR
    DownloadData.py --> Train.py
    Train.py --> Evaluation.py
    ```

  * If you just want to check current models reports:

    ```mermaid
    graph LR
    DownloadData.py --> Evaluation.py
    ```






## Some Parameters Values of Models

> The following tables are recommended hyperparameters of the current data set(in folder `Data/all_data.pkl`) and models(in folder `Saved_Models`). If you want to regenerate the training or test set for training some new models, these values may need to be changed.

* ViT

|              | Batch Size | Epochs | Learning Rate |
| ------------ | ---------- | ------ | ------------- |
| Normal       | 16         | 15     | 0.0000008     |
| Diabetes     | 16         | 15     | 0.00000002    |
| Glaucoma     | 16         | 15     | 0.00000001    |
| Cataract     | 16         | 15     | 0.00000001    |
| Age_related  | 16         | 15     | 0.00000001    |
| Hypertension | 16         | 15     | 0.0000000034  |
| Pathological | 16         | 15     | 0.00000001    |
| Other        | 16         | 15     | 0.00000003    |

 

* ResNet50

|              | Batch Size | Epochs | Learning Rate |
| ------------ | ---------- | ------ | ------------- |
| Normal       | 2          | 15     | 0.000002      |
| Diabetes     | 2          | 15     | 0.00000005    |
| Glaucoma     | 2          | 15     | 0.00000005    |
| Cataract     | 2          | 15     | 0.0000001     |
| Age_related  | 2          | 15     | 0.00000005    |
| Hypertension | 2          | 15     | 0.000000007   |
| Pathological | 2          | 15     | 0.000000007   |
| Other        | 2          | 15     | 0.00000004    |