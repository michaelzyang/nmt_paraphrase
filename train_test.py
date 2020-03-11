import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime


def train(train_loader, val_loader, n_epochs, model, criterion, optimizer, scheduler=None,
          save_path="./", start_epoch=1, device='gpu', task='Classification'):
    raise NotImplementedError
    # if save_path[-1] != '/':
    #     save_path = save_path + '/'
    #
    # model.train()
    #
    # print(f"Beginning training at {datetime.now()}")
    # if start_epoch == 1:
    #     with open(save_path + f"results.txt", mode='a') as f:
    #         f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    #
    # for epoch in range(start_epoch, n_epochs + 1):
    #     avg_loss = 0.0
    #     for batch_num, (feats, labels) in enumerate(train_loader):
    #         feats, labels = feats.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = model(feats)[1]
    #         loss = criterion(outputs, labels.long())
    #         loss.backward()
    #         optimizer.step()
    #
    #         avg_loss += loss.item()
    #
    #         if batch_num % 400 == 399:
    #             print(f'Epoch: {epoch}\t\
    #                   Batch: {batch_num + 1}\t\
    #                   Avg-Loss: {avg_loss / 400:.4f}\t\
    #                   {datetime.now()}')
    #             avg_loss = 0.0
    #
    #         torch.cuda.empty_cache()
    #         del feats
    #         del labels
    #         del loss
    #
    #     if task == 'Classification':
    #         val_loss, val_acc = test_classify(model, val_loader, criterion, device)
    #         train_loss, train_acc = test_classify(model, train_loader, criterion, device)
    #         print(f'Train Loss: {train_loss:.4f}\t\
    #               Train Accuracy: {train_acc:.4f}\t\
    #               Val Loss: {val_loss:.4f}\t\
    #               Val Accuracy: {val_acc:.4f}\t\
    #               {datetime.now()}')
    #         if scheduler:
    #             scheduler.step(val_loss)
    #     else:
    #         raise NotImplementedError("Verification evaluation not implemented.")
    #
    #     # save epoch
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': train_loss,
    #         'train_acc': train_acc,
    #         'val_loss': val_loss,
    #         'val_acc': val_acc
    #     }, save_path + f"checkpoint_{epoch}_{val_acc}.pth")
    #
    #     with open(save_path + f"results.txt", mode='a') as f:
    #         f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc}\n")


def test_classify(model, test_loader, criterion, device='gpu'):
    raise NotImplementedError
    # model.eval()
    # test_loss = []
    # accuracy = 0
    # total = 0
    # with torch.no_grad():
    #     for batch_num, (feats, labels) in enumerate(test_loader):
    #         feats, labels = feats.to(device), labels.to(device)
    #         outputs = model(feats)[1]
    #
    #         _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
    #         pred_labels = pred_labels.view(-1)
    #
    #         loss = criterion(outputs, labels.long())
    #
    #         accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
    #         total += len(labels)
    #         batch_size = feats.size()[0]  # final batch might be smaller than the rest
    #         test_loss.extend([loss.item()] * batch_size)
    #
    #         torch.cuda.empty_cache()
    #         del feats
    #         del labels
    #
    # model.train()
    # return np.mean(test_loss), accuracy / total


def eval(test_loader, model, device='gpu', out_type="classes"):
    raise NotImplementedError
    # if out_type == "classes":
    #     out_idx = 1
    # elif out_type == "embeddings":
    #     out_idx = 0
    # else:
    #     raise ValueError(f"out_type must be 'classes' or 'embeddings'. {out_type} given.")
    # model.eval()
    #
    # results = []
    # with torch.no_grad():
    #     for i, (feats, _) in enumerate(test_loader):
    #         feats = feats.to(device)
    #
    #         outputs = model(feats)[out_idx]
    #         outputs = outputs.detach().to("cpu")
    #         if out_type == "classes":
    #             _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
    #             outputs = pred_labels.view(-1)
    #         else: # out_type == "embeddings"
    #             if i % 50 == 49:
    #                 print(f"Completed batch {i + 1} at {datetime.now()}")
    #
    #         results.append(outputs)
    #
    #         torch.cuda.empty_cache()
    #         del feats
    #
    # return torch.cat(results)
