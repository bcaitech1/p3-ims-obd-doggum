import time
import argparse
import random
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd

from datasets.data_loader import setup_loader
from utils.utils import fast_hist, label_accuracy_score, AverageMeter
from datasets.data_loader import CustomDataLoader
from transforms.Augmentations import TestAugmentation, CustomAugmentation, CustomAugmentation2
from visualize.showplots import showImageMask
from network.utils import get_model
from loss.optimizer import get_optimizer
from loss.utils import get_loss


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seed', type=int, default=21)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--criterion', type=str, default='cross_entropy')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--model', type=str, default='unetmnv2')

# Meta Pseudo label
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')


# Container environment
parser.add_argument("--dataset_path", type=str, default= '../input/data')

args = parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# https://github.com/kekmodel/MPL-pytorch
def main():
    """
    Main Function
    :return:
    """
    print(args)

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

    seed_everything(args.seed)
    sorted_df = setup_loader(args)

    # train.json / validation.json / test.json 디렉토리 설정
    train_path = args.dataset_path + '/train.json'
    val_path = args.dataset_path + '/val.json'
    test_path = args.dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_transform = CustomAugmentation()

    val_transform = CustomAugmentation()

    test_transform = TestAugmentation()
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
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn,
                                               drop_last = True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=2,
                                              collate_fn=collate_fn)

    # showImageMask(train_loader, list(sorted_df.Categories))
    # showImageMask(val_loader, list(sorted_df.Categories))
    # showImageMask(test_loader, list(sorted_df.Categories), test=True)

    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test

    teacher_model = get_model(args, num_classes=12)
    student_model = get_model(args, num_classes=12)

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    if not args.eval:
        criterion = get_loss(args)
        t_optimizer = get_optimizer(args, teacher_model)
        s_optimizer = get_optimizer(args, student_model)

        teacher_model.zero_grad()
        student_model.zero_grad()

        # model_path = f'./saved/{args.model}_best_model.pt'
        #
        # # best model 불러오기
        # checkpoint = torch.load(model_path, map_location=device)
        # teacher_model.load_state_dict(checkpoint)
        # student_model.load_state_dict(checkpoint)

        train(args, args.epochs, teacher_model, student_model, train_loader, val_loader, test_loader, criterion, t_optimizer, s_optimizer, saved_dir, val_every, device)

    return
    # best model 저장된 경로
    model_path = f'./saved/{args.model}_best_model.pt'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    model.eval()
    showImageMask(test_loader, list(sorted_df.Categories), test=True, model=model, device=device)

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
    submission.to_csv(f"./submission/{args.model}_{time_string}.csv", index=False)


def save_model(model, saved_dir, file_name='SegNet_best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def train(args, num_epochs, teacher_model, student_model, data_loader, val_loader, test_loader, criterion, t_optimizer, s_optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_loss = 9999999
    best_mIoU = 0

    labeled_epoch = 0
    unlabeled_epoch = 0

    labeled_iter = iter(data_loader)
    unlabeled_iter = iter(test_loader)

    for step in tqdm(range(1, num_epochs*381)):
        teacher_model.train()
        student_model.train()

        try:
            images_l, masks_l, _ = labeled_iter.next()
        except:
            labeled_epoch += 1
            labeled_iter = iter(data_loader)
            images_l, masks_l, _ = labeled_iter.next()

        try:
            images_u, image_infos_u = unlabeled_iter.next()
        except:
            unlabeled_epoch += 1
            unlabeled_iter = iter(test_loader)
            images_u, image_infos_u = unlabeled_iter.next()

        images_l = torch.stack(images_l)
        masks_l = torch.stack(masks_l).long()
        images_u = torch.stack(images_u)
        images_l, masks_l = images_l.to(device), masks_l.to(device)
        images_u = images_u.to(device)

        batch_size = images_l.shape[0]
        t_images = torch.cat((images_l, images_u))
        t_logits = teacher_model(t_images)
        t_logits_l = t_logits[:batch_size]
        t_logits_u = t_logits[batch_size:]
        del t_logits

        t_loss_l = criterion(t_logits_l, masks_l)

        # print("\nt_loss_u")
        # print(t_logits_u.shape)
        # print(t_logits_u[0])
        soft_pseudo_label = torch.softmax(t_logits_u.detach()/args.temperature, dim=-1)
        # print("\nsoft_pseudo_label")
        # print(soft_pseudo_label.shape)
        # print(soft_pseudo_label[0])
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=1)
        # print("\nmax_probs")
        # print(max_probs.shape)
        # print(max_probs[0])
        # print("\nhard_pseudo_label")
        # print(hard_pseudo_label.shape)
        # print(hard_pseudo_label[0])
        mask = max_probs.ge(args.threshold).float()
        # print("\nmask")
        # print(mask.shape)
        # print(mask[0])
        t_loss_u = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_logits_u, dim=1)).sum(dim=1) * mask
        )
        weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        s_images = torch.cat((images_l, images_u))
        s_logits = student_model(s_images)
        s_logits_l = s_logits[:batch_size]
        s_logits_u = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), masks_l)
        s_loss = criterion(s_logits_u, hard_pseudo_label)

        s_optimizer.zero_grad()
        s_loss.backward()
        s_optimizer.step()

        with torch.no_grad():
            s_logits_l = student_model(images_l)
        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), masks_l)
        dot_product = s_loss_l_old - s_loss_l_new
        _, hard_pseudo_label = torch.max(t_logits_u.detach(), dim=1)
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_u, hard_pseudo_label)
        t_loss = t_loss_uda + t_loss_mpl

        t_loss.backward()
        t_optimizer.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if step % 25 == 0:
            print(step, labeled_epoch, unlabeled_epoch)
            print(t_loss.item())
            print(s_loss.item())
            print()

        if step % 381 == 0:
            validation(step, teacher_model, val_loader, criterion, device)
            validation(step, student_model, val_loader, criterion, device)

    return



def validation(epoch, model, data_loader, criterion, device):
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

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            # 각각의 mask에 대한 confusion matrix를 hist에 저장
            for lt, lp in zip(outputs, masks.detach().cpu().numpy()):
                hist += fast_hist(lt.flatten(), lp.flatten(), n_class)

        avrg_loss = total_loss / cnt
        mIoU = label_accuracy_score(hist)
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss,
                                                                          mIoU))

    return avrg_loss

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
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array

if __name__ == '__main__':
    main()