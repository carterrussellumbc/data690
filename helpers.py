import os
import shutil
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib as mpl
import matplotlib.pyplot as plt


# train_df = pd.read_csv(data_dir)
# cancer_positive = train_df[train_df['label'] == 1]
# cancer_negative = train_df[train_df['label'] == 0]

class Helpers:

    def __init__(self):
        self.data_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\train_labels.csv"

    # def plot_roc_curve(self, fpr, tpr):
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.plot(fpr, tpr)
    #     plt.xlabel('False positive rate')
    #     plt.ylabel('True positive rate')
    #     plt.title('ROC curve')
    #     plt.legend(loc='best')
    #     print('hi :)')
    #     plt.savefig('C:\\Users\\Carter\\git\\data690_project_2\\roc_curve.jpg')

    def generate_roc_curve_data(self, model, val_ds, train_ds):
        X = []
        Y = []
        for images, labels in val_ds:
            for image in images:
                # X.append(image)  # append tensor
                X.append(image.numpy())  # append numpy.array
                # X.append(image.numpy().tolist())  # append list
            for label in labels:
                # Y.append(label)  # append tensor
                Y.append(label.numpy())  # append numpy.array
                # Y.append(label.numpy().tolist())  # append list
        test_images_x = X
        test_images_y = Y
        y_pred = model.predict(val_ds)
        fpr_neg, tpr_neg, thresholds_neg = roc_curve(test_images_y, y_pred[:, 0])
        fpr_pos, tpr_pos, thresholds_pos = roc_curve(test_images_y, y_pred[:, 1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_neg, tpr_neg)
        plt.plot(fpr_pos, tpr_pos)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig('roc_curve.jpg')

    def move_images_into_folder_classes(self):
        '''
        A one-off function to move the cancer photos to two separate directories named cancer_positive and cancer_negative for use by tensorflow.
        This function is not scalable and I would make it better if it were not a one-off function
        '''
        root_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data"
        source_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\train"
        for picture in cancer_negative['id']:
            picture = picture + ".tif"
            if picture.lower().endswith(('.tif')):
                image_name, ext = os.path.splitext(picture)
                folder_path = os.path.join(root_dir, "cancer_negative")

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                source_path = os.path.join(source_dir, picture)
                destination_path = os.path.join(folder_path, picture)
                shutil.move(source_path, destination_path)

    def convert_tif_to_jpg(self):
        '''
        Converts the cancer .tif images to cancer .jpg images
        It's a really bad function that doesn't scale but I would make it better if it wasn't just a one-off function
        '''
        cancer_positive_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data\\cancer_positive"
        cancer_positive_jpg_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data\\jpg\\cancer_positive"
        cancer_negative_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data\\cancer_negative"
        cancer_negative_jpg_dir = "C:\\Users\\Carter\\Downloads\\histopathologic-cancer-detection\\data\\jpg\\cancer_negative"
        if not os.path.exists(cancer_negative_jpg_dir):
            os.makedirs(cancer_negative_jpg_dir)
        for picture in os.listdir(cancer_negative_dir):
            tif_img = Image.open(f"{cancer_negative_dir}\\{picture}")
            jpg_img = tif_img.convert("RGB")
            jpg_img.save(f"{cancer_negative_jpg_dir}\\{picture.split('.')[0]}.jpg")