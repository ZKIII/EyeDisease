from torch.utils.data import DataLoader
from DataTransform import *
from Models import *


def train_loss_acc(model, train_loader, device, criterion, optimizer):
    """
    Train and evaluate the model loss and accuracy.
    :param model: ViT or ResNet model
    :param train_loader: training set loaded by DataLoader
    :param device: cpu or cuda
    :param criterion: binary cross entropy loss function
    :param optimizer: optimizer function Adam
    :return: train_loss, train_acc
    """
    total_loss = 0
    total_acc = 0
    for x_img, y in train_loader:
        x_img, y = x_img.to(device), y.to(device)
        pred_train = model(x_img) # prediction of current model
        train_loss = criterion(pred_train, y.float()) # current training loss
        acc = ((pred_train > 0.5) == y.float()).sum().item() / y.size(0) # current training accuracy
        total_loss += train_loss.item() # cumulative training loss
        total_acc += acc # cumulative training accuracy
        optimizer.zero_grad() # calculated the gradient
        train_loss.backward() # backpropagation
        optimizer.step()
    return total_loss / len(train_loader), total_acc / len(train_loader)


def test_loss_acc(model, test_loader, device, criterion):
    """
    Evaluate the model testing loss and accuracy.
    :param model: ViT or ResNet model
    :param test_loader: testing set loaded by DataLoader
    :param device: cpu or cuda
    :param criterion: binary cross entropy loss function
    :param optimizer: optimizer function Adam
    :return: test_loss, test_acc
    """
    total_loss = 0
    total_acc = 0
    for x_img, y in test_loader:
        x_img, y = x_img.to(device), y.to(device)
        pred_test = model(x_img) # prediction of current model
        test_loss = criterion(pred_test, y.float()) # current test loss
        acc = ((pred_test > 0.5) == y.float()).sum().item() / y.size(0) # current testing accuracy
        total_loss += test_loss.item() # cumulative testing loss
        total_acc += acc # cumulative testing accuracy
    return total_loss / len(test_loader), total_acc / len(test_loader)


def train(model_name, device, train_load, test_load, epochs, lr):
    """
    Train model and show the evaluations.
    :param model_name: str, "ViT" or "ResNet"
    :param device: cpu or cuda
    :param train_load: training set loaded by DataLoader
    :param test_load: testing set loaded by DataLoader
    :param epochs: int, number of epochs
    :param lr: float, learning rate
    :return: best selected model, train_loss, train_acc, test_loss, test_acc
    """
    # create model by model_name
    model = None
    if model_name == 'ViT':
        model = ViT().to(device)
    elif model_name == 'ResNet50':
        model = ResNet().to(device)
    criterion = nn.BCELoss().to(device) # binary cross-entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam gradient

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    best_model = None
    prev_acc = None
    for e in range(epochs):
        model.train() # train model

        # add current training and testing loss and accuracy
        curr_train_loss, curr_train_acc = train_loss_acc(model, train_load, device, criterion, optimizer)
        curr_test_loss, curr_test_acc = test_loss_acc(model, test_load, device, criterion)
        train_losses.append(curr_train_loss)
        test_losses.append(curr_test_loss)
        train_acc.append(curr_train_acc)
        test_acc.append(curr_test_acc)

        print(f"Epoch {e + 1}: Train Loss: {curr_train_loss:.4f}, Train Acc: {curr_train_acc:.4f}, Test Loss: {curr_test_loss:.4f}, Test Acc: {curr_test_acc:.4f}")

        # best selected by the lowest testing loss
        if prev_acc is None or curr_test_acc >= prev_acc:
            prev_acc = curr_test_acc
            best_model = model

    return best_model, train_losses, test_losses, train_acc, test_acc


def save_model(model_name, device, data, y_classes, index, batch_size, lr, epoches):
    """
    Save the trained model into the Saved_Models folder.
    Save the losses and accuracy of the trained models into the Losses_Acc folder.
    :param model_name: str, "ViT" or "ResNet"
    :param device: cpu or cuda
    :param data: the original data set
    :param y_classes: the target class name
    :param index: the column index of the target class
    :param batch_size: int, the batch size of training set
    :param lr: int, the learning rate
    :param epoches: int, number of epochs
    :return:
    """
    print(f"----------Start Train Model: {model_name}, y: {y_classes}------------")
    train_set = DataTransformSet(data.x_imgs_train, data.y_train[:, index].reshape(data.y_train.shape[0], 1))
    test_set = DataTransformSet(data.x_imgs_test, data.y_test[:, index].reshape(data.y_test.shape[0], 1))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    best_model, train_losses, test_losses, train_acc, test_acc = train(model_name, device, train_loader, test_loader, epoches, lr)

    with open(f"Losses_Acc/{model_name}_{y_classes}_losses_acc.pkl", "wb") as loss_acc_file:
        pickle.dump((train_losses, test_losses, train_acc, test_acc), loss_acc_file)

    torch.save(best_model, f"Saved_Models/{model_name}_{y_classes}.pth")
    torch.cuda.empty_cache()
    print(f"----------End Train Model: {model_name}, y: {y_classes}------------\n")


def all_models(y_vit, y_resnet, vit_batch_size, resnet_batch_size, epochs):
    """
    Train all models.
    :param y_vit: dict, ViT target class name: learning rate
    :param y_resnet: dict, ResNet target class name: learning rate
    :param vit_batch_size: int, the batch size of training set for ViT
    :param resnet_batch_size: int, the batch size of training set for ResNet
    :return:
    """
    data = DataTransform()
    data.load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, k in enumerate(y_vit.keys()):
        save_model("ViT", device, data, k, i, vit_batch_size, y_vit[k], epochs)
        save_model("ResNet50", device, data, k, i, resnet_batch_size, y_resnet[k], epochs)

# -----------------------------------------------------------------------------------
# The following function is used for training a signle model, which is
# easier to find a suitable combination of batch size, learning rate and epochs.

# def single_model():
#     data = DataTransform()
#     data.load_data()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     save_model("ResNet50", device, data, "Other", 7, 2, 0.00000004, 15)
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    # single_model()

    epochs = 15
    vit_batch_size = 16
    y_vit = {"Normal": 0.0000008,
             "Diabetes": 0.00000002,
             "Glaucoma": 0.00000001,
             "Cataract": 0.00000005,
             "Age_related": 0.00000001,
             "Hypertension": 0.0000000034,
             "Pathological": 0.00000001,
             "Other": 0.00000003}

    resnet_batch_size = 2
    y_resnet = {"Normal": 0.000002,
                "Diabetes": 0.00000005,
                "Glaucoma": 0.00000005,
                "Cataract": 0.0000001,
                "Age_related": 0.00000005,
                "Hypertension": 0.000000007,
                "Pathological": 0.000000007,
                "Other": 0.00000004}

    all_models(y_vit, y_resnet, vit_batch_size, resnet_batch_size, epochs)