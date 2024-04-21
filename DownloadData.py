import os
import zipfile

def main():
    loss_acc_folder = os.path.join(os.getcwd(), 'Losses_Acc')
    if not os.path.exists(loss_acc_folder):
        os.makedirs(loss_acc_folder)

    reports_folder = os.path.join(os.getcwd(), 'Reports')
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    print("Download Data...")
    os.system("gdown https://drive.google.com/uc?id=1HTOOXIrf4iFd88u2gdCI7jPovKhSGRYx")
    print("Unzip Data Folder...")
    data = zipfile.ZipFile("Data.zip", 'r')
    for d in data.namelist():
        data.extract(d, "")
    data.close()

    print("Download Saved_Models...")
    os.system("gdown https://drive.google.com/uc?id=1862QUN49PRLHCAuLBFew8U9YBJVN8sqw")
    print("Unzip Saved_Models Folder...")
    saved_models = zipfile.ZipFile("Saved_Models.zip", 'r')
    for s in saved_models.namelist():
        saved_models.extract(s, "")
    saved_models.close()

    if os.path.exists("Data.zip"):
        os.remove("Data.zip")

    if os.path.exists("Saved_Models.zip"):
        os.remove("Saved_Models.zip")


if __name__ == "__main__":
    main()