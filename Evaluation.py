from Train import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def plot_loss_acc(model_name, y_class):
    """
    Plot and save loss and accuracy plots by current "{model_name}_{y_class}_losses_acc.pkl" file.
    :param model_name: str, current model name
    :param y_class: str, current class label
    :return:
    """
    loss_acc_path = f"Losses_Acc/{model_name}_{y_class}_losses_acc.pkl"
    with (open(loss_acc_path, "rb") as loss_acc_file):
        train_losses, test_losses, train_acc, test_acc = pickle.load(loss_acc_file)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, label="Train Loss")
        ax1.plot(test_losses, label="Test Loss")
        ax1.set_title(f"Train and Test Loss of {model_name} of {y_class}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(train_acc, label="Train Accuracy")
        ax2.plot(test_acc, label="Test Accuracy")
        ax2.set_title(f"Train and Test Accuracy of {model_name} of {y_class}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        plt.savefig(f"Reports/{model_name}_{y_class}_Loss_Acc.png")
        plt.clf()


def plot_all(model_names, y_classes):
    """
    Plot and save all plots.
    :param model_names: list, all model names
    :param y_classes: list, all class labels
    :return:
    """
    for model_name in model_names:
        for y_class in y_classes:
            plot_loss_acc(model_name, y_class)


def acc(model_name, data, y_class, index, device):
    """
    Show the accuracy by current "{model_name}_{y_class}.pth" file.
    :param model_name: str, current model name
    :param data: DataTransform class data
    :param y_class: str, current class label
    :param index: int, current class label index
    :param device: cpu or cuda
    :return: accuracy, predictions
    """
    # load test sets
    test_set = DataTransformSet(data.x_imgs_test, data.y_test[:, index].reshape(data.y_test.shape[0], 1))
    test_loader = DataLoader(test_set, batch_size=1)

    # load model
    model_path = f"Saved_Models/{model_name}_{y_class}.pth"
    model = torch.load(model_path)

    total_acc = 0
    pred_test_lst = []
    for x_img, y in test_loader:
        x_img, y = x_img.to(device), y.to(device)
        pred_test = model(x_img)
        acc = ((pred_test > 0.5) == y.float()).sum().item() / y.size(0)  # current testing accuracy
        total_acc += acc  # cumulative testing accuracy

        if pred_test.detach().cpu().numpy()[0][0] > 0.5:
            pred_test_lst.append(1)
        else:
            pred_test_lst.append(0)
    pred_test_array = np.array(pred_test_lst).T
    return total_acc / len(test_loader), pred_test_array


def all_acc_f1_score(model_names, y_classes):
    """
    Show all accuracy and f1 score.
    :param model_names: list, all model names
    :param y_classes: list, all class labels
    :return:
    """
    # load data and set device
    data = DataTransform()
    data.load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("Reports/Acc_F1.txt", "w") as acc_f1_file:
        for model_name in model_names:
            for y_class in y_classes:
                index = y_classes.index(y_class)
                curr_acc, pred_test_array = acc(model_name, data, y_class, index, device)
                curr_f1 = f1_score(data.y_test[:, index], pred_test_array, average='macro') # f1-score of targets and predictions
                results = f"{model_name}_{y_class} | Accuracy: {curr_acc:.4f}, F1: {curr_f1:.4f}"
                print(results)
                acc_f1_file.write(f'{results}\n')


if __name__ == '__main__':
    model_names = ["ViT", "ResNet50"]
    y_classes = ["Normal", "Diabetes", "Glaucoma", "Cataract", "Age_related", "Hypertension", "Pathological", "Other"]
    plot_all(model_names, y_classes)
    all_acc_f1_score(model_names, y_classes)