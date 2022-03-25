import pandas as pd
import matplotlib.pyplot as plt
# epoch, lr, valid_loss, dice_c
# col_list = ["epochs", "lr","loss", "dice", "jaccard"]

def get_metrics(validation_path, train_path):
    validation_df = pd.read_csv(validation_path)
    train_df = pd.read_csv(train_path)

    loss_val = validation_df["loss"].tolist()
    loss_train = train_df["loss"].tolist()


    dc_val = validation_df["dice"].tolist()
    dc_train = train_df["dice"].tolist()

    epochs = validation_df["epoch"].size

    iou_val = validation_df["iou"].tolist()
    iou_train = train_df["iou"].tolist()
    return {
        "epochs":epochs,
        "loss_val":loss_val, 
        "loss_train":loss_train, 
        "dc_val":dc_val, 
        "dc_train":dc_train, 
        "iou_val":iou_val, 
        "iou_train":iou_train
    }

def plot_results(label1, title, **data):
    print (data.items())
    # epochs = range(1,x)
    # plt.plot(epochs, y1, 'b', label=label1)
    # plt.plot(epochs, y2, 'g', label=label2)
    # plt.title(title)
    # plt.xlabel('Epochs')

    # plt.ylabel(title)

    # plt.legend()

    # plt.show()

def plot_boxplot(label2="", title="", **data):
    # print(data)
    fig,ax = plt.subplots()
    c="gray"
    ax.boxplot(
        data.values(),
        patch_artist=True,
        boxprops=dict(facecolor=c, color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color="black"),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color="white")
        )

    ax.set_xticklabels(data.keys())


dice_loss_val_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\256_200_epochs_DiceLoss_Normalization_mean\\result_validation.csv"
dice_loss_train_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\256_200_epochs_DiceLoss_Normalization_mean\\result_train.csv"

logits_loss_val_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\BCE_LOGITS_200_2\\result_validation.csv"
logits_loss_train_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\BCE_LOGITS_200_2\\result_train.csv"

dicebce_loss_val_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\200_256_DiceBCE_Normalization_mean\\result_validation.csv"
dicebce_loss_train_path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\AttU_Net\\200_256_DiceBCE_Normalization_mean\\result_train.csv"

dice = get_metrics(dice_loss_val_path, dice_loss_train_path)
dice_bce = get_metrics(dicebce_loss_val_path,dicebce_loss_train_path)
bcelogits = get_metrics(logits_loss_val_path,logits_loss_train_path)

# plot_results(len(epochs)+1, validation_loss, train_loss, label1="Validation", label2= "Training", title="Training and Validation loss")
# plot_results(len(epochs)+1, valication_dice, train_dice, label1="Validation", label2= "Training", title= "Dice")
# plot_results(len(epochs)+1, valication_jaccard, train_jaccard, label1="Validation", label2= "Training", title="Jaccard")

plot_boxplot("Validation","Dice", Dice = dice["dc_val"][70:], BCE_Dice = dice_bce["dc_val"][70:], BCE_Logits=bcelogits["dc_val"][70:])


plt.show()