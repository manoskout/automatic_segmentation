import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def draw_contours(img_mri, true_mask, predicted):
    """
    This function draws both the predicted mask and ground truth mask
    The green contours are the predicted
    The blue contours are the ground truth
    """
    contours_pred = getContours(predicted)
    contours_true = getContours(true_mask)
    with_pred = cv2.drawContours(img_mri, contours_pred,-1,(0,255,0),1)
    combined_img = cv2.drawContours(with_pred, contours_true,-1,(255,0,0),1)
    
    return combined_img


def getContours(im):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours


def main():
    path= config.results_path
    num_of_epochs = np.arange(1,81)
    pred = f"{config.filename}{1}_result_PRED.jpg"
    ground_truth = f"{config.filename}{1}_result_GT.jpg"
    img = f"{config.filename}{1}_result_INPUT.jpg"

    predicted = cv2.imread(os.path.join(path,pred))
    img_mri = cv2.imread(os.path.join(path,img))
    true_mask = cv2.imread(os.path.join(path,ground_truth))
    new_img = draw_contours(img_mri, true_mask, predicted)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    axcolor = 'yellow'
    ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03],facecolor=axcolor)
    slider = Slider(ax_slider, 'Images', 1, 80, valinit=1)
    img = ax.imshow(new_img)
    slider.on_changed(update)
    plt.show()

def update(val):
    path= config.results_path
    pred = f"{config.filename}{int(val)}_result_PRED.jpg"
    ground_truth = f"{config.filename}{int(val)}_result_GT.jpg"
    img = f"{config.filename}{int(val)}_result_INPUT.jpg"
    print(int(val))
    predicted = cv2.imread(os.path.join(path,pred))
    img_mri = cv2.imread(os.path.join(path,img))
    true_mask = cv2.imread(os.path.join(path,ground_truth))
    new_img = draw_contours(img_mri, true_mask, predicted)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    fig.canvas.draw_idle()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--results_path', type=str, default="C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\U_Net\\Multiorgan_4_Batch_DiceLoss")
    parser.add_argument('--filename', type=str, default='U_Net_valid_')
    fig, ax = plt.subplots()
    # fig.canvas.manager.set_window_title('UNet Binary')

    config = parser.parse_args()
    main()
