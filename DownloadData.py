import os
import zipfile

def main():
    reports_folder = os.path.join(os.getcwd(), 'Reports')
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    print("Download Data...")
    os.system("gdown https://drive.google.com/uc?id=1HTOOXIrf4iFd88u2gdCI7jPovKhSGRYx")
    print("Unzip Data Folder...")
    if os.path.exists("Data.zip"):
        data = zipfile.ZipFile("Data.zip", 'r')
        for d in data.namelist():
            data.extract(d, "")
        data.close()    

    print("Download Saved_Models...")
    os.system("gdown https://drive.google.com/uc?id=1862QUN49PRLHCAuLBFew8U9YBJVN8sqw")
    print("Unzip Saved_Models Folder...")
    if os.path.exists("Saved_Models.zip"):
        saved_models = zipfile.ZipFile("Saved_Models.zip", 'r')
        for s in saved_models.namelist():
            saved_models.extract(s, "")
        saved_models.close()

    print("Download Losses_Acc...")
    os.system("gdown https://drive.google.com/uc?id=163qWhbsJfH-VzBk4CEhzCaC93RwWtc_Z")
    print("Unzip Losses_Acc Folder...")
    if os.path.exists("Losses_Acc.zip"):
        losses_acc = zipfile.ZipFile("Losses_Acc.zip", 'r')
        for l in losses_acc.namelist():
            losses_acc.extract(l, "")
        losses_acc.close()

    if os.path.exists("Data.zip"):
        os.remove("Data.zip")

    if os.path.exists("Saved_Models.zip"):
        os.remove("Saved_Models.zip")

    if os.path.exists("Losses_Acc.zip"):
        os.remove("Losses_Acc.zip")


if __name__ == "__main__":
    main()
