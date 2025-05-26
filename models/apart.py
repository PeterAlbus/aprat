import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import AdapterVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from torch.nn import functional as F
import torch.nn as nn
import clip

num_workers = 8

class FeatureProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc(x)
    
def calculate_distances(features):
    """计算特征之间的欧氏距离"""
    features = F.normalize(features, dim=1)  # 归一化特征
    dot_product = torch.matmul(features, features.T)  # 计算点积
    dot_product = (1 - dot_product) / 2  # 计算余弦距离
    return dot_product


def average_features_per_class(features, targets):
    """对每个类别的特征取平均值"""
    unique_labels = torch.unique(targets).cpu()
    class_features = []
    class_labels = []
    for label in unique_labels:
        mask = targets == label
        if mask.any():  # 确保该类别在当前batch中有样本
            class_feature = features[mask].mean(dim=0)  # 计算该类别的平均特征
            class_features.append(class_feature)
            class_labels.append(label.item())
    class_features = torch.stack(class_features)  # 将列表堆叠成张量
    return class_features

def compute_text_similarity(text_features):
    """ 计算文本特征之间的相似度（例如余弦相似度）"""
    text_features = F.normalize(text_features, p=2, dim=-1)  # 归一化
    similarity_matrix = torch.matmul(text_features, text_features.T)  # 计算相似度矩阵
    return similarity_matrix

def compute_similarity_matrix(features, temperature=0.07):
    """
    计算特征之间的余弦相似度矩阵
    features: 输入的特征，大小为 (batch_size, feature_dim)
    temperature: 温度系数，用于缩放相似度
    """
    features = F.normalize(features, p=2, dim=-1)  # 归一化
    similarity_matrix = torch.matmul(features, features.T) / temperature
    return similarity_matrix


def relation_based_loss(image_features, text_features, targets, temperature=0.07):
    """
    基于图像和文本特征的相似度矩阵计算的损失函数
    image_features: 图像特征，size (batch_size, feature_dim)
    text_features: 文本特征，size (batch_size, feature_dim)
    targets: 样本标签，size (batch_size,)
    margin: 三元组损失中的间隔
    temperature: 温度系数
    """

    # 计算图像和文本的相似度矩阵
    image_similarity_matrix = compute_similarity_matrix(image_features, temperature)
    text_similarity_matrix = compute_similarity_matrix(text_features, temperature)

    # 损失函数是两个相似度矩阵之间的均方误差 (MSE)
    loss = F.mse_loss(image_similarity_matrix, text_similarity_matrix)
    return loss


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)   
        self._network = AdapterVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.theta = args["theta"] if args["theta"] is not None else 100
        self.args = args

        self.clip_model, _ = clip.load("ViT-B/32", device=self._device)
        # 假设图像特征和文本特征的维度分别为 512 和 256，目标投影维度为 128
        # self.image_projection_head = FeatureProjectionHead(input_dim=64, output_dim=64).to(device) # CIFAR100 ResNet32
        # self.image_projection_head = FeatureProjectionHead(input_dim=512, output_dim=64).to(device) # Food101/ImageNetSubset ResNet18
        self.image_projection_head = FeatureProjectionHead(input_dim=1024, output_dim=64).to(self._device) # ViT (vit_l_32)
        # self.image_projection_head = FeatureProjectionHead(input_dim=768, output_dim=64).to(device) # swin_t
        # self.image_projection_head = FeatureProjectionHead(input_dim=192, output_dim=64).to(device) # CIFAR100 - ViT Tiny 
        self.text_projection_head = FeatureProjectionHead(input_dim=512, output_dim=64).to(self._device)

        self.class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def project_features(self, image_features, text_features):
        text_features = text_features.to(torch.float32)
        print(f"Image features: {image_features}")
        image_projection = self.image_projection_head(image_features)  # 投影图像特征
        text_projection = self.text_projection_head(text_features)  # 投影文本特征
        return image_projection, text_projection       

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.class_order = data_manager._class_order
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self.args["imbalance"]:
            cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]: # true
            optimizer = self.get_optimizer()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        net_param = [p for name, p in self._network.backbone.named_parameters() if  p.requires_grad]
        
        base_param =  [p for name, p in self._network.backbone.named_parameters() if 'pool' in name and p.requires_grad]
        fc_param =  [p for name, p in self._network.backbone.named_parameters() if 'pool' not in name and p.requires_grad]
        
        param_f = {'params': base_param, 'lr': self.init_lr * 0.1, 'weight decay': self.weight_decay}
        param_s = {'params': fc_param, 'lr': self.init_lr, 'weight decay': self.weight_decay}
        param = [param_f, param_s]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                param
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["init_cls"]  != self.args["increment"]:
            self.args["scheduler"] = "cosine"
        else:
            self.args["scheduler"] = "constant"

        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["pool_size"]
            prev_end = task_id * args["pool_size"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["pool_size"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["pool_size"]
            prev_end = task_id * args["pool_size"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["pool_size"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                weight = cls_num_list[targets]

                output, image_features = self._network(inputs, task_id=self._cur_task, train=True, weight=weight, return_features=True) 
                 # 使用CLIP模型获取标签特征
                labels = [self.class_names[self.class_order[t.item()]] for t in targets]
                text_descriptions = [f"A photo of a {label}" for label in labels]
                # 去重
                # text_descriptions = list(set(text_descriptions))
                text_tokens = clip.tokenize(text_descriptions).to(self._device)
                text_features = self.clip_model.encode_text(text_tokens)
                # print(f"Text features shape: {text_features.shape}")
                # print(f"Image features shape: {image_features.shape}")
                mapped_image_features, mapped_text_features = self.project_features(image_features, text_features)
                # 计算基于文本关系的损失
                loss_feature = relation_based_loss(mapped_image_features, mapped_text_features, targets, temperature=0.7)
                logging.info(f"Loss feature: {loss_feature.item()}")
                pool = output["pool_id"]
                logits = output["logits"]
                logits = logits[:, self._known_classes : self._total_classes] 
                fake_targets = targets - self._known_classes
                loss1 = F.cross_entropy(logits, fake_targets.long())
                loss = loss1

                
                logits_few = output["logits_few"]
                logits_few = logits_few[:, self._known_classes : self._total_classes] 
                logits_all = logits + logits_few
                target = F.one_hot(fake_targets, self._total_classes - self._known_classes)
                
                loss_all = F.cross_entropy(logits_all, fake_targets.long())
                loss += loss_all
                loss_few = - (pool.squeeze(1) * (target * torch.log(nn.Softmax(dim=-1)(logits_few)+1e-7)).sum(dim=1)).sum()
                loss += loss_few
                loss /= 3  
                    
                weight_norm = torch.where(weight <= self.theta, 1.0, 0.1) 
                weight_weight = torch.where(weight <= self.theta, 10.0, 1.0)
                    
                if epoch < 5:
                    match_loss = ((weight_norm - pool.squeeze(1))**2 * weight_weight).sum()
                else:
                    match_loss  = ((1.0 - pool.squeeze(1)) ** 2 * pool.squeeze(1) * 10).sum()

                loss += match_loss

                if self.args["pull_constraint"] and 'reduce_sim' in output:  
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim'] 
                    loss -= self.args["pull_constraint_coeff"] * output['reduce_sim_few']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits_all, dim=1)

                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy(pool1 {:.2f}, pool2 {:.2f}, pool_all {:.2f})".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc[0],test_acc[1], test_acc[2]
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                res = self._network(inputs, task_id=self._cur_task)
                outputs = res["logits"][:, :self._total_classes]
                outputs += res["logits_few"][:, :self._total_classes] 
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct1,correct2, correct3, total = 0, 0, 0, 0
        label_list = []
        cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                weight = cls_num_list[targets]
                label_list.append(weight.cpu())
                res = model(inputs, task_id=self._cur_task, weight=weight) 
                outputs1 = res["logits"][:, :self._total_classes]
                outputs2= res["logits_few"][:, :self._total_classes] 
                outputs3 = outputs1 + outputs2
                
            predicts1 = torch.max(outputs1, dim=1)[1]
            correct1 += (predicts1.cpu() == targets).sum()

            predicts2 = torch.max(outputs2, dim=1)[1]
            correct2 += (predicts2.cpu() == targets).sum()

            predicts3 = torch.max(outputs3, dim=1)[1]
            correct3 += (predicts3.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct1) * 100 / total, decimals=2), np.around(tensor2numpy(correct2) * 100 / total, decimals=2), np.around(tensor2numpy(correct3) * 100 / total, decimals=2)
