import copy
import numpy as np
import os
from PIL import Image
import pandas as pd
import sys
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch
from torchsummary import summary
from torch.utils import data
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.nn.functional as F

sys.path.insert(0,'../') 
sys.path.insert(0,'../my_models') 
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)

from raug.loader import get_data_loader 
from raug.train import fit_model
from raug.eval import test_model
from aug_isic import ImgTrainTransform, ImgEvalTransform
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.utils.loader import get_labels_frequency

from my_model import set_model, get_activation_fn, get_activations, ConvReg
from kd_losses import D_KD, DR


class MyDataset_ss (data.Dataset):

    def __init__(self, imgs_path, labels, meta_data=None, transform=None, train=True):
        super().__init__()
        self.imgs_path = imgs_path
        self.labels = labels
        self.meta_data = meta_data
        self.train = train

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()


    def __len__(self):
        return len(self.imgs_path)


    def __getitem__(self, item):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(self.imgs_path[item]).convert("RGB")
        image = np.array(image) 
        if self.train:
            if np.random.rand() < 0.5:
                image = image[:,::-1,:]

        image0 = np.rot90(image, 0).copy()
        image0 = Image.fromarray(image0)
        image0 = self.transform(image0)

        image1 = np.rot90(image, 1).copy()
        image1 = Image.fromarray(image1)
        image1 = self.transform(image1)

        image2 = np.rot90(image, 2).copy()
        image2 = Image.fromarray(image2)
        image2 = self.transform(image2)

        image3 = np.rot90(image, 3).copy()
        image3 = Image.fromarray(image3)
        image3 = self.transform(image3)

        img = torch.stack([image0, image1, image2, image3])

        img_id = self.imgs_path[item].split('/')[-1].split('.')[0]

        if self.meta_data is None:
            meta_data = []
        else:
            meta_data = self.meta_data[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return img, labels, meta_data, img_id


def get_data_loader_ss (imgs_path, labels, meta_data=None, transform=None, batch_size=30, shuf=True, num_workers=4,
                     pin_memory=True, train=True):
    dt = MyDataset_ss(imgs_path, labels, meta_data, transform, train)
    dl = data.DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory)
    return dl


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def norm(x):

    n = np.linalg.norm(x)
    return x / n


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():

    # Dataset variables
    _folder = 2
    _base_path = "../../data/ISIC2019"
    _csv_path_train = os.path.join(_base_path, "train", "ISIC2019_parsed_train.csv")
    _imgs_folder_train = os.path.join(_base_path, "train", "imgs")

    _csv_path_test = os.path.join(_base_path, "test", "ISIC2019_parsed_test.csv")
    _imgs_folder_test = os.path.join(_base_path, "test", "imgs")

    _use_meta_data = False 
    _neurons_reducer_block = 0
    _comb_method = None 
    _comb_config = None 
    _batch_size = 32 
    _epochs_tune = 60
    _epochs = 150 
    
    # Training variables
    _best_metric = "loss"
    _pretrained = True
    _lr_init = 0.001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15 
    _weights = "frequency"
    _use_wce = True # default as True, Sept.10.21

    _model_name_teacher = 'resnet-50'
    _model_name_student = 'mobilenet'

    _kd_method = 'ssd_kd'    

    _layer_s = 'feats_7x7'
    _layer_t = 'feats_7x7'
    _lambd = 0.5
    _lambd_drkd = 1
    _lambd_crkd = 1000  
    _hr_w = 1
    
    _save_folder = "results/" + _kd_method + "_t_" + _model_name_teacher + "_s_" + _model_name_student + "_fold_" + \
                    str(_folder) + '_drkd_' + str(_lambd_drkd) + '_crkd_' + str(_lambd_crkd)  + '_hrw_' + str(_hr_w) #+ "_" + str(time.time()).replace('.', '')

    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _csv_path_train, _imgs_folder_train, _csv_path_test, _imgs_folder_test, _lr_init, _sched_factor,
          _sched_min_lr, _sched_patience, _batch_size, _epochs,_epochs_tune,  _early_stop, _weights, 
          _model_name_teacher, _model_name_student,_kd_method,
           _pretrained, _save_folder, _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data,
           _layer_s, _layer_t, _lambd, _lambd_drkd, _lambd_crkd, _hr_w, _use_wce):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")  
    
    meta_data_columns = ['age_approx', 'female', 'male', 'anterior torso', 'head/neck', "lateral torso",
                         'lower extremity', 'oral/genital', 'palms/soles', 'posterior torso',  'upper extremity']

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    
    _checkpoint_best_base = os.path.join(_save_folder, 'best-checkpoint')
    os.makedirs(_checkpoint_best_base, exist_ok=True)
    _checkpoint_best = os.path.join(_checkpoint_best_base, 'best-checkpoint.pth')

    _checkpoint_last_base = os.path.join(_save_folder, 'last-checkpoint')
    os.makedirs(_checkpoint_last_base, exist_ok=True)
    _checkpoint_last = os.path.join(_checkpoint_last_base, 'last-checkpoint.pth')    

    logs_path_base = os.path.join(_save_folder, "logs")
    os.makedirs(logs_path_base, exist_ok=True)
    logs_path = os.path.join(logs_path_base, "output.txt")
    sys.stdout=open(logs_path, "w")

    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]
    
    
    ####################################################################################################################

    # Loading validation data
    val_imgs_id = val_csv_folder['image'].values
    val_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder['diagnostic_number'].values
    if _use_meta_data:
        val_meta_data = val_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        val_meta_data = None
    val_data_loader = get_data_loader_ss (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, 
                                       pin_memory=True, train=False)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    # Loading training data
    print("- Loading training data...")
    train_imgs_id = train_csv_folder['image'].values
    train_imgs_path = ["{}/{}.jpg".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['diagnostic_number'].values
    if _use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        train_meta_data = None
    train_data_loader = get_data_loader_ss (train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, 
                                       pin_memory=True, train=True)  #16
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))


    # Loading test data
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test['image'].values
    test_imgs_path = ["{}/{}.jpg".format(_imgs_folder_test, img_id) for img_id in test_imgs_id]
    test_labels = csv_test['diagnostic_number'].values
    csv_test['lateral torso'] = 0
    if _use_meta_data:
        test_meta_data = csv_test[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "test_pred"),
        'pred_name_scores': 'predictions.csv',
    }
    test_data_loader = get_data_loader(test_imgs_path, test_labels, test_meta_data, transform=ImgEvalTransform(),
                                        batch_size=_batch_size, shuf=False, num_workers=16, 
                                        pin_memory=True)   
    
    print("-"*50)
    ####################################################################################################################

    ser_lab_freq = get_labels_frequency(train_csv_folder, "diagnostic", "image")
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)   

    ####################################################################################################################

    def load_ckpt (checkpoint_path, model):
        if not os.path.exists(checkpoint_path):
            raise Exception ("The {} does not exist!".format(checkpoint_path))

        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()

        return model
    
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()


    def get_trainable_list(model_name, model, layer_name, device='cuda:0'):
        
        trainable_list = nn.ModuleList([])
        trainable_list.append(model)      

        model.eval()

        x_ = torch.randn(1,3,224, 224).to(device)
        out_ = get_activations(model_name, model, x_)

        feat_dim = out_[layer_name].shape[-1]

        proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            )
        
        proj_head = proj_head.to(device)
        trainable_list.append(proj_head)

        return trainable_list
          

    ### Teacher model: load/freeze pretrained model
    print("- Loading", _model_name_teacher)
    model_t_ckpt_path = "results/" + str(_comb_method) + "_" + _model_name_teacher + \
                        "_fold_" + str(_folder) + "/best-checkpoint/best-checkpoint.pth"

    ckpt_ss_t_path_base = os.path.join(_save_folder, 'teacher_ckpt_ss')  
    os.makedirs(ckpt_ss_t_path_base, exist_ok=True)
    ckpt_ss_t_path = os.path.join(ckpt_ss_t_path_base, 'best-teacher-ss.pth')

    net_t = set_model(_model_name_teacher, len(_labels_name), pretrained=_pretrained, 
                        neurons_reducer_block=_neurons_reducer_block)

    print("- Loading pretrained models...")
    model_t = load_ckpt(model_t_ckpt_path, net_t)
    model_t = model_t.to(device)
    trainable_list_t = get_trainable_list(model_name=_model_name_teacher, model=model_t, layer_name='avg_pool')

    feat_fn_t = get_activation_fn(_model_name_teacher)


    ##############################  Fine-tuning teacher model's SS module ##############################
    model_t.train()
    optimizer_t = optim.SGD([{'params':trainable_list_t[0].parameters(), 'lr':0.0},
                             {'params':trainable_list_t[1].parameters(), 'lr':_lr_init}],
                             momentum=0.9, weight_decay=0.001)

    scheduler_lr_t = MultiStepLR(optimizer_t, milestones=[30, 45], gamma=0.1)        
 
    print("- Starting fine-tuning teacher's SS module...")
    print('-'*50)

    early_stop_count = 0
    best_loss = 100000.0
    t_epoch = _epochs_tune 
    model_t_ss_best = copy.deepcopy(trainable_list_t[1].state_dict())
    for epoch in range(t_epoch):
        model_t.eval()
        trainable_list_t[1].train()

        loss_record = AverageMeter()
        acc_record = AverageMeter()

        start = time.time()
        ############################# SS module train #############################
        for data_batch in train_data_loader:
            x, _, _, _ = data_batch
            optimizer_t.zero_grad()

            x = x.to(device)
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)
            activations = feat_fn_t(model_t, x)
            tmp = activations['avg_pool']
            rep = trainable_list_t[1](tmp)

            batch = int(x.size(0) / 4)

            nor_index = (torch.arange(4*batch) % 4 == 0).to(device)
            aug_index = (torch.arange(4*batch) % 4 != 0).to(device)

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)  

            loss.backward()
            optimizer_t.step()

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            loss_record.update(loss.item(), 3*batch)
            acc_record.update(batch_acc.item(), 3*batch)


        run_time = time.time() - start
        info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}(s)\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
        epoch+1, t_epoch, run_time, loss_record.avg, acc_record.avg)

        print(info)

        ############################# SS module validation #############################
        model_t.eval()
        trainable_list_t[1].eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()        

        for data_batch_val in val_data_loader:
            x, _, _, _ = data_batch_val
            x = x.to(device)
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            with torch.no_grad():
                activations = feat_fn_t(model_t, x)
                tmp = activations['avg_pool']
                rep = trainable_list_t[1](tmp)

            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).to(device)
            aug_index = (torch.arange(4*batch) % 4 != 0).to(device)

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(),3*batch)
            loss_record.update(loss.item(), 3*batch)

        run_time = time.time() - start

        info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}(s)\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
                epoch+1, t_epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)


        early_stop_count += 1

        if loss_record.avg < best_loss:
            best_loss = loss_record.avg
            early_stop_count = 0
            model_t_ss_best = copy.deepcopy(trainable_list_t[1].state_dict())
            torch.save(model_t_ss_best, ckpt_ss_t_path)
        
        scheduler_lr_t.step()   

        if early_stop_count >= _early_stop:
            print("The early stop trigger was activated.")
            break

    print()

    
    ######## loading best SS checkpoint
    print("- Loading", _model_name_teacher, "'s SS module..")
    net_ss_t = trainable_list_t[1]
    ckpt_ss_t = torch.load(ckpt_ss_t_path)
    net_ss_t.load_state_dict(ckpt_ss_t)
   
    freeze_model(model_t)
    freeze_model(trainable_list_t[1])
    model_t.eval()
    trainable_list_t[1].eval()

    ############################### Student model Setup ###############################
    print("- Loading", _model_name_student) 
    model_s = set_model(_model_name_student, len(_labels_name), 
                            pretrained=_pretrained, neurons_reducer_block=_neurons_reducer_block)

    model_s = model_s.to(device)
    trainable_list_s = get_trainable_list(model_name=_model_name_student, model=model_s, layer_name='avg_pool')
    feat_fn_s = get_activation_fn(_model_name_student)

    models = {'model_s': model_s, 'model_t': model_t}
    feat_fn = {'feat_fn_s': feat_fn_s, 'feat_fn_t': feat_fn_t}
    _layers = {'_layer_s': _layer_s, '_layer_t': _layer_t}

    #### get feat map dimension
    model_t.eval()
    model_s.eval()
    x_ = torch.randn(1,3,224, 224).to(device)
    out_s = feat_fn_s(model_s, x_, student=False)
    out_t = feat_fn_t(model_t, x_, student=False)

    regress_s = ConvReg(out_s[_layer_s].shape, out_t[_layer_t].shape)   
    trainable_list_s.append(regress_s)


    optimizer = optim.SGD(trainable_list_s.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience) 
    if _use_wce:
        wce_weight=torch.Tensor(_weights).to(device)
    else:
        wce_weight=None
    loss_wce = nn.CrossEntropyLoss(weight=wce_weight).to(device)
    loss_hr = DR(weight=wce_weight, _layers=_layers, module_list=trainable_list_s, lambd=_lambd, 
                    lambd_rkd=_lambd_drkd, lambd_crkd=_lambd_crkd).to(device)
    ######################################### Start training student ######################################## 
    print("- Starting the training phase...")
    print("-" * 50)

    start = time.time()
    kd_T = 4.0 
    tf_T = 4.0
    ratio_tf = 1.0
    ratio_ss = 0.75
    ss_T = 0.5
    ce_weight = 0.1
    kd_weight = 0.9
    tf_weight = 2.7
    ss_weight = 10.0
    best_loss = 100000.
    early_stop_count = 0
    best_epoch = 0
    
    ############################### Train student model epochs ###############################
    for epoch in range(_epochs):

        model_s.train()  
        trainable_list_s[1].train()
        trainable_list_s[2].train()

        loss1_record = AverageMeter()
        loss2_record = AverageMeter()
        loss3_record = AverageMeter()
        loss4_record = AverageMeter()
        loss5_record = AverageMeter()
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()

        ############################### Train student model epoch-by-epoch ###############################
        for data_batch in train_data_loader:
            x, target, _, _ = data_batch
            optimizer.zero_grad()

            x = x.to(device)
            target = target.to(device)

            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).to(device)
            aug_index = (torch.arange(4*batch) % 4 != 0).to(device)        
        
            with torch.no_grad():
                activations_t = feat_fn_t(model_t, x)
                tmp_t = activations_t['avg_pool']
                t_feat = trainable_list_t[1](tmp_t)
                knowledge = activations_t['logits']
                nor_knowledge = F.softmax(knowledge[nor_index] / kd_T, dim=1)
                aug_knowledge = F.softmax(knowledge[aug_index] / tf_T, dim=1)

            activations_s = feat_fn_s(model_s, x)
            tmp_s = activations_s['avg_pool']
            s_feat = trainable_list_s[1](tmp_s)
            output = activations_s['logits']
            log_nor_output = F.log_softmax(output[nor_index] / kd_T, dim=1)
            log_aug_output = F.log_softmax(output[aug_index] / tf_T, dim=1)

            ## error level ranking
            aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().to(device)

            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)

            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * ratio_tf)
            index = index[:correct_num+wrong_keep]
            distill_index_tf = torch.sort(index)[0]

            s_nor_feat = s_feat[nor_index]
            s_aug_feat = s_feat[aug_index]
            s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

            t_nor_feat = t_feat[nor_index]
            t_aug_feat = t_feat[aug_index]
            t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

            t_simi = t_simi.detach()
            aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(t_simi, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * ratio_ss)
            index = index[:correct_num+wrong_keep]
            distill_index_ss = torch.sort(index)[0]     


            s_nor_feat = s_feat[nor_index]
            s_aug_feat = s_feat[aug_index]
            s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)    


            t_simi = t_simi.detach()
            aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(t_simi, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * ratio_ss)
            index = index[:correct_num+wrong_keep]
            distill_index_ss = torch.sort(index)[0]    

            log_simi = F.log_softmax(s_simi / ss_T, dim=1)
            simi_knowledge = F.softmax(t_simi / ss_T, dim=1)

            loss1 = loss_wce(output[nor_index], target)
            loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * kd_T * kd_T
            loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                            reduction='batchmean') * tf_T * tf_T
            loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                            reduction='batchmean') * ss_T * ss_T    
            loss5 = loss_hr(activations_s, activations_t)  

            loss = ce_weight * loss1 + kd_weight * loss2 + tf_weight * loss3 + ss_weight * loss4 + \
                    _hr_w * loss5 

            loss.backward()
            optimizer.step()

            cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
            ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), len(distill_index_ss))

            loss5_record.update(loss5.item(), batch)

            cls_acc_record.update(cls_batch_acc.item(), batch)
            ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)

        run_time = time.time() - start
        info = 'student_train_Epoch:{:d}/{:d}\t run_time:{:.3f}(s)\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
        epoch+1, _epochs, run_time, loss1_record.avg, loss2_record.avg, cls_acc_record.avg)

        print(info)

        model_s.eval()

        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        ############################### Validation student model epochs ###############################
        for data_batch in val_data_loader:

            x, target, _, _ = data_batch
            x = x[:,0,:,:,:].to(device)
            target = target.to(device)

            with torch.no_grad():      
                output = model_s(x)
                # loss = F.cross_entropy(output, target)
                loss = loss_wce(output, target)

            batch_acc = accuracy(output, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))

        run_time = time.time() - start
        early_stop_count += 1

        if loss_record.avg < best_loss:
            best_loss = loss_record.avg
            best_epoch = epoch
            early_stop_count = 0
            info_to_save = {
            'epoch': best_epoch,
            'model_state_dict':  model_s.state_dict(),             
            }
            torch.save(info_to_save, _checkpoint_best)

        scheduler_lr.step(best_loss)

        info = 'student_test_Epoch:{:d}/{:d}\t run_time:{:.3f}(s)\t cls_acc:{:.3f}\t val loss:{:.3f}\t max to stop:{:d}\t best@epoch: {:d}\n'.format(
                epoch+1, _epochs, run_time, acc_record.avg, loss_record.avg, early_stop_count, best_epoch)
        print(info)
        
        if early_stop_count >= _early_stop:
            print("The early stop trigger was activated.")
            break
        
    
    info_to_save_last = {
            'epoch': epoch,
            'model_state_dict':  model_s.state_dict(),             
    }
    
    torch.save(info_to_save_last, _checkpoint_last)

    model_s = load_ckpt(_checkpoint_best, model_s)
    model_s.eval()
   

    # ########################################### Test student model ##########################################################

    # Testing the test partition
    print("- Evaluating the test partition...")
    
    test_model (model_s, test_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_wce, save_pred=True,
                partition_name='test', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    
    torch.cuda.empty_cache()
    sys.stdout.close()
    # ####################################################################################################################

