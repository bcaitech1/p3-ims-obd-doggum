import time
import torch
import numpy as np

from utils import label_accuracy_score, add_hist

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print('Start training..')
    best_miou = 0
    for epoch in range(num_epochs):
        now = time.time()
        hist = np.zeros((12, 12))
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
    
            # inference
#             outputs = model(images)['out']
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            # step 주기에 따른 loss, mIoU 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mIoU: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item(), mIoU))
        one_epochs_time = time.time() - now
        print('Epoch [{}/{}], time: {:.4f}'.format(epoch+1, num_epochs, one_epochs_time))
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_miou = validation(epoch + 1, model, val_loader, criterion, device)
            if val_miou > best_miou:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_miou = val_miou
                save_model(model, saved_dir, epoch)
                best_opt = optimizer.state_dict()
            else:
                file_path = saved_dir + "/best_miou.pt"
                model.load_state_dict(torch.load(file_path, map_location=device))
                optimizer.load_state_dict(best_opt)
                for g in optimizer.param_groups:
                    g['lr']=g['lr']/3
                
#         scheduler.step()


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((12, 12))
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

#             outputs = model(images)['out']
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mIoU))

    return avrg_loss, mIoU