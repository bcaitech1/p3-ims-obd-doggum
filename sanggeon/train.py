import time
import argparse
import random
import os
import warnings
from importlib import import_module
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

from datasets.data_loader import setup_loader
from utils.utils import add_hist, label_accuracy_score
from datasets.data_loader import CustomDataLoader
from visualize.showplots import showImageMask
from network.utils import get_model
from loss.optimizer import get_optimizer
from loss.utils import get_loss
from config import Config
from transforms.Augmentations import invTrans


def get_config():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation3')
    parser.add_argument('--criterion', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unetmnv2')
    parser.add_argument('--continue_load', type=str, default='')
    parser.add_argument('--eval_load', type=str, default='')

    # Container environment
    parser.add_argument("--dataset_path", type=str, default= '../input/data')

    args = parser.parse_args()

    config = Config(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        eval=args.eval,
        augmentation=args.augmentation,
        criterion=args.criterion,
        optimizer=args.optimizer,
        model=args.model,
        continue_load=args.continue_load,
        eval_load=args.eval_load,
        dataset_path=args.dataset_path
    )

    return config


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def main(config):
    """
    Main Function
    :return:
    """
    print(config)

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

    seed_everything(config.seed)
    sorted_df = setup_loader(config)

    # train.json / validation.json / test.json 디렉토리 설정
    train_path = config.dataset_path + '/train.json'
    val_path = config.dataset_path + '/val.json'
    test_path = config.dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    augmentation_module = getattr(import_module("transforms.Augmentations"), config.augmentation)
    # train_transform = get_augmentation(config, mode='train')
    train_transform = augmentation_module(mode='train')

    val_transform = augmentation_module(mode='val')

    test_transform = augmentation_module(mode='test')
    # from albumentations.pytorch import ToTensorV2
    # train_transform = A.Compose([
    #     ToTensorV2()
    # ])
    #
    # val_transform = A.Compose([
    #     ToTensorV2()
    # ])
    #
    # test_transform = A.Compose([
    #     ToTensorV2()
    # ])

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path, sorted_df=sorted_df, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, sorted_df=sorted_df, mode='val', transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, sorted_df=sorted_df, mode='test', transform=test_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               collate_fn=collate_fn,
                                               drop_last = True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              num_workers=2,
                                              collate_fn=collate_fn)

    showImageMask(train_loader, list(sorted_df.Categories))
    showImageMask(val_loader, list(sorted_df.Categories))
    # showImageMask(test_loader, list(sorted_df.Categories), test=True)
    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test

    model = get_model(config, num_classes=12)
    # x = torch.randn([1, 3, 512, 512])
    # print("input shape : ", x.shape)
    # out = model(x).to(device)
    # print("output shape : ", out.size())

    model = model.to(device)

    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    if config.eval_load == '':
        criterion = get_loss(config)
        optimizer = get_optimizer(config, model)

        if config.continue_load in ('miou', 'loss'):

            if config.continue_load == 'miou':
                # best model 저장된 경로
                model_path = f'./saved/{config.model}_best_model_miou.pt'
            elif config.contiue_load == 'loss':
                # best model 저장된 경로
                model_path = f'./saved/{config.model}_best_model_loss.pt'

            # best model 불러오기
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)


        train(config.epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)

    if config.eval_load == 'miou':
        # best model 저장된 경로
        model_path = f'./saved/{config.model}_best_model_miou.pt'
    elif config.eval_load == 'loss':
        # best model 저장된 경로
        model_path = f'./saved/{config.model}_best_model_loss.pt'
    else:
        print('must choice miou / loss')


    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    model.eval()
    showImageMask(test_loader, list(sorted_df.Categories), test=True, model=model, device=device)

    # 그냥 쌩으로 모델 불러온거면 테스트해보기.
    if config.eval_load != '':
        criterion = get_loss(config)
        validation(0, model, val_loader, criterion, device, crf=True)
        return

    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    # submission.csv로 저장
    tm = time.gmtime()
    time_string = time.strftime('%yy%mm%dd_%H_%M_%S', tm)
    submission.to_csv(f"./submission/{config.model}_{time_string}.csv", index=False)


def save_model(model, saved_dir, file_name='SegNet_best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_loss = 9999999
    best_mIoU = 0
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # inference
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, step + 1, len(data_loader), loss.item()))

        # if (epoch + 1) == 6:
        #     save_model(model, saved_dir, f'{config.model}_best_model6.pt')
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, avrg_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir, f'{config.model}_best_model_loss.pt')

            if avrg_mIoU > best_mIoU:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = avrg_mIoU
                save_model(model, saved_dir, f'{config.model}_best_model_miou.pt')



'''
# Default Values are
apperance_kernel = [8, 164, 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [3, 10]         # PairwiseGaussian  [sxy, compat] 

# or if you want to to specify seprately for each XY direction and RGB color channel then

apperance_kernel = [(1.5, 1.5), (64, 64, 64), 100] # PairwiseBilateral [sxy, srgb, compat]  
spatial_kernel = [(0.5, 0.5), 10]                  # PairwiseGaussian  [sxy, compat] 
'''
# https://www.programcreek.com/python/example/106424/pydensecrf.densecrf.DenseCRF2D
h, w = 512, 512


def dense_crf(probs, img=None, n_classes=12, n_iters=10, scale_factor=1):
    c, h, w = probs.shape

    if img is not None:
        assert (img.shape[1:3] == (h, w))
        img = np.transpose(img, (1, 2, 0)).copy(order='C')
        img = np.uint8(255 * img)

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(3, 3), compat=10)
    d.addPairwiseBilateral(sxy=10, srgb=5, rgbim=np.copy(img), compat=10)
    Q = d.inference(n_iters)

    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
    return preds

def validation(epoch, model, data_loader, criterion, device, crf:bool=False):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        n_class = 12
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):

            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, height, width)

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            if crf == False:
                outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            else:
                probs_array = []
                for image, prob in zip(images, outputs):
                    prob = F.softmax(prob, dim=0)
                    prob = dense_crf(img=np.around(invTrans(image).cpu().numpy()).astype(float),
                                     probs=prob.cpu().numpy())
                    probs_array += [np.argmax(prob, axis=0)]

                outputs = np.array(probs_array)

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss,
                                                                          mIoU))

    return avrg_loss, mIoU

import albumentations as A
def test(model, data_loader, device):
    size = 256

    transform = A.Compose([A.Resize(256, 256)])

    print('Start prediction.')
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(data_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))

            probs_array = []
            for image, prob in zip(imgs, outs):
                prob = F.softmax(torch.from_numpy(prob), dim=0)
                prob = dense_crf(img=np.around(InvNormalize(image).cpu().numpy()).astype(float), probs=prob.cpu().numpy())
                probs_array += [np.argmax(prob, axis=0)]

            oms = np.array(probs_array)

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = np.around(oms.reshape([oms.shape[0], size * size]).astype(int))
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array

if __name__ == '__main__':
    config = get_config()
    main(config)