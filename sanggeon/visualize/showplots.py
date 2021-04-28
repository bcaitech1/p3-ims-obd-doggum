
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = False

def showImageMask(data_loader, category_names, test:bool=False):
    # data_loader의 output 결과(image 및 mask) 확인
    if not test:
        for imgs, masks, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs
            temp_masks = masks

            break
    else:
        for imgs, image_infos in data_loader:
            image_infos = image_infos[0]
            temp_images = imgs

            break

    if test:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    print('image shape:', list(temp_images[0].shape))
    if not test:
        print('mask shape: ', list(temp_masks[0].shape))
        print('Unique values, category of transformed mask : \n',
              [{int(i), category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])

    ax1.imshow(temp_images[0].permute([1, 2, 0]))
    ax1.grid(False)
    ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize=15)

    if not test:
        ax2.imshow(temp_masks[0])
        ax2.grid(False)
        ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize=15)

    plt.show()