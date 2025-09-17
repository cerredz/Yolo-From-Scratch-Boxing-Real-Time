import torch

def loss(predictions, targets, lambda_coord=5, S=7, B=2, C=20):

    l1, l2, l3, l4, l5 = 0, 0, 0, 0, 0

    # calculate the first loss term (x and y)
    for pred in predictions:
        s_idx_1, s_idx_2 = -1, -1
        for i in range(0, S ** 2):
            s_idx_2 += 1
            if s_idx_2 == S:
                s_idx_2 = 0
                s_idx_1 += 1
            for j in range(0, B):
                x_idx = C + (j * 5):C + 1 + (j * 5)
                y_idx = C + 1 + (j * 5):C + 2 + (j * 5)

                loss_x_squared = pow((targets[:,s_idx_1,s_idx_2,x_idx] - predictions[:,s_idx_1,s_idx_2,x_idx], 2))
                loss_y_squared = pow((targets[:,s_idx_1,s_idx_2,y_idx] - predictions[:,s_idx_1,s_idx_2,y_idx], 2))
                l1 += (loss_x_squared + loss_y_squared)

    l1 *= lambda_coord

    # calculate the second loss term (w and h)
    for i in range(0, S ** 2):




