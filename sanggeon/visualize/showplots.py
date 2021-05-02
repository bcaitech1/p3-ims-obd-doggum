
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms

plt.rcParams['axes.grid'] = False

def showImageMask(data_loader, category_names, test:bool=False, model=None, device=None):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])

    # data_loader의 output 결과(image 및 mask) 확인
    if not test:
        # train 이나 val
        for imgs, masks, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs
            temp_masks = masks

            break
    elif test and model != None:
        # 훈련된 model로 test data 예측 확인
        for imgs, image_infos in data_loader:
            image_infos = image_infos
            temp_images = imgs

            model.eval()
            # inference
            outs = model(torch.stack(temp_images).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            break
    else:
        # 그냥 test data 잘 불러오나 테스트하는거.
        for imgs, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs

            break

    if test and model == None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    if test and model != None:
        i = 3
    else:
        i = 0
    print('image shape:', list(temp_images[i].shape))
    if not test:
        print('mask shape: ', list(temp_masks[i].shape))
        print('Unique values, category of transformed mask : \n',
              [{int(i), category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])

    ax1.imshow(invTrans(temp_images[i]).permute([1, 2, 0]))
    ax1.grid(False)
    if model == None:
        ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize=15)
    else:
        ax1.set_title("input image : {}".format(image_infos[i]['file_name']), fontsize=15)

    if test and model != None:
        ax2.imshow(oms[i])
        ax2.grid(False)
        ax2.set_title("Predicted : {}".format(image_infos[i]['file_name']), fontsize=15)
    elif not test:
        ax2.imshow(temp_masks[0])
        ax2.grid(False)
        ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize=15)

    plt.show()