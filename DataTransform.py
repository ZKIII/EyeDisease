import numpy as np
import pandas as pd
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DataTransform:
    """
    Split and save image, features and labels into a single file for future training and testing.
    """
    def __init__(self, path="Data"):
        """
        Initialize the DataTransform class.
        :param path: string, path of the data folder
        """
        self.path = path
        self.x_features_train = None
        self.x_imgs_train = None
        self.y_train = None
        self.x_features_test = None
        self.x_imgs_test = None
        self.y_test = None

    def __str__(self):
        """
        Print the training and testing data set dimensions.
        :return: string representation of the DataTransform class
        """
        x_features_train_shape = "The shape of x_features_train: {}\n".format(self.x_features_train.shape)
        x_imgs_train_shape = "The shape of x_imgs_train: {}\n".format(self.x_imgs_train.shape)
        y_train_shape = "The shape of y_train: {}\n".format(self.y_train.shape)
        x_features_test_shape = "The shape of x_features_test: {}\n".format(self.x_features_test.shape)
        x_imgs_test_shape = "The shape of x_imgs_test: {}\n".format(self.x_imgs_test.shape)
        y_test_shape = "The shape of y_test: {}\n".format(self.y_test.shape)
        return (x_features_train_shape + x_imgs_train_shape + y_train_shape +
                x_features_test_shape + x_imgs_test_shape + y_test_shape)

    def generate_data(self, img_size=224, train_size=0.9, random = True):
        """
        Generate the training(90%) and testing(10%) data set.
        :param img_size: int, size of the image(224, 224, 3)
        :param train_size: float, size of the training set
        :return:
        """
        # read csv data as pd.DataFrame
        dataset_path = self.path + "/dataset.csv"
        imgs_path = self.path + "/train"
        if random == True:
            data = pd.read_csv(dataset_path).sample(frac=1).reset_index(drop=True)
        else:
            data = pd.read_csv(dataset_path)

        # training and test data set split index
        split_index = round(len(data) * train_size)

        # open all images and resize to (224, 224, 3) and save into np.array as uint8 format
        x_features = data[["Patient Age", "Female", "Male", "Left", "Right"]].values
        x_imgs = data["Fundus"].values.tolist()
        for i in range(len(x_imgs)):
            img_path = imgs_path + "/" + x_imgs[i]
            img = Image.open(img_path)
            img = img.resize((img_size, img_size))  # Resize the image
            x_imgs[i] = np.array(img, dtype=np.uint8)  # Convert to uint8 to save memory
        x_imgs = np.array(x_imgs)

        # features training and testing sets
        self.x_features_train = x_features[:split_index, :]
        self.x_features_test = x_features[split_index:, :]

        # images training and testing sets
        self.x_imgs_train = x_imgs[:split_index, :, :, :]
        self.x_imgs_test = x_imgs[split_index:, :, :, :]

        # labels training and testing sets
        y_features = ["Normal", "Diabetes", "Glaucoma", "Cataract", "Age_related",
                      "Hypertension", "Pathological", "Other"]
        y = data[y_features].values
        self.y_train = y[:split_index, :]
        self.y_test = y[split_index:, ]

    def save_data(self):
        """
        Save the training and testing data set as pickle file.
        :return:
        """
        with open(self.path + "/all_data.pkl", "wb") as data_file:
            pickle.dump((self.x_features_train, self.x_imgs_train, self.y_train,
                         self.x_features_test, self.x_imgs_test, self.y_test), data_file)

    def load_data(self):
        """
        Load the training and testing data set from pickle file.
        :return:
        """
        with (open(self.path + "/all_data.pkl", "rb") as data_file):
            self.x_features_train, self.x_imgs_train, self.y_train, \
                self.x_features_test, self.x_imgs_test, self.y_test = pickle.load(data_file)


class DataTransformSet(Dataset):
    """
    A Dataset structure of Pytorch.
    """
    def __init__(self, x_imgs, y):
        """
        Initialize the DataTransformSet class by images and labels.
        :param x_imgs: array, images data
        :param y: array, labels data
        """
        self.x_imgs = x_imgs
        self.y = y
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((224, 224)),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])

    def __len__(self):
        """
        Return the number of samples in the dataset.
        :return: int, the number of samples
        """
        return len(self.x_imgs)

    def __getitem__(self, index):
        """
        Return a sample of data by index
        :param index: int, index of the sample
        :return: tuple, a sample of images and labels
        """
        return self.transform(self.x_imgs[index]), self.y[index]


if __name__ == "__main__":
    data = DataTransform()
    print("Generating data...")
    data.generate_data()
    print("Start saving data...")
    data.save_data()
    print("Done")
    print(data)
