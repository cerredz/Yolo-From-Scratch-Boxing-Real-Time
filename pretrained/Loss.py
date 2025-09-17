import torch

def loss(predictions, targets, lambda_coord=5, S=7, B=2, C=20):

    l1 = 0
    for i in range(0, S ** 2):
        for j in range(0, B):
            x_idx = 20 + (j * 5):21 + (j * 5)
            y_idx = 21 + (j * 5):22 + (j * 5)

            loss_x_squared = pow((targets[:,:,:,x_idx] - predictions[:,:,:,x_idx], 2))
            loss_y_squared = pow((targets[:,:,:,y_idx] - predictions[:,:,:,y_idx], 2))
            l1 += (loss_x_squared + loss_y_squared)



