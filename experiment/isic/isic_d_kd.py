import copy
import os
import pandas as pd
import sys
import time
import torch.optim as optim
import torch.nn as nn
import torch
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn.functional as F

sys.path.insert(0,'../') 
sys.path.insert(0,'../my_models') 

from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from aug_isic import ImgTrainTransform, ImgEvalTransform
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.utils.loader import get_labels_frequency
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.train_kd import fit_model_kd
from raug.eval_kd import test_model_kd

from my_model import set_model, get_activation_fn, get_activations, ConvReg
from kd_losses import D_KD


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
    _batch_size = 128 
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

    _kd_method = 'd_kd'    

    _layer_s = 'feats_7x7'
    _layer_t = 'feats_7x7'
    _lambd = 0.5
    _lambd_drkd = 1.0
    _lambd_crkd = 1000
    
    _save_folder = "results/" + _kd_method + "_t_" + _model_name_teacher + "_s_" + _model_name_student + "_fold_" + \
                    str(_folder) + '_drkd_' + str(_lambd_drkd) + '_crkd_' + str(_lambd_crkd) #+ "_" + str(time.time()).replace('.', '')

    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _csv_path_train, _imgs_folder_train, _csv_path_test, _imgs_folder_test, _lr_init, _sched_factor,
          _sched_min_lr, _sched_patience, _batch_size, _epochs, _early_stop, _weights, 
          _model_name_teacher, _model_name_student,_kd_method,
           _pretrained, _save_folder, _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data,
           _layer_s, _layer_t, _lambd, _lambd_drkd, _lambd_crkd, _use_wce):

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
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

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
    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
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
    train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, transform=ImgTrainTransform(),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)  #16
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
                                        batch_size=_batch_size, shuf=False, num_workers=16, pin_memory=True)   
    
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


    ### Teacher model: load/freeze pretrained model
    print("- Loading", _model_name_teacher)
    model_t_ckpt_path = "results/" + str(_comb_method) + "_" + _model_name_teacher + \
                        "_fold_" + str(_folder) + "/best-checkpoint/best-checkpoint.pth"

    net_t = set_model(_model_name_teacher, len(_labels_name), pretrained=_pretrained, 
                        neurons_reducer_block=_neurons_reducer_block)

    print("- Loading pretrained models...")
    model_t = load_ckpt(model_t_ckpt_path, net_t)
    freeze_model(model_t) 

    model_t = model_t.to(device)
    feat_fn_t = get_activation_fn(_model_name_teacher)

    ### Student model
    print("- Loading", _model_name_student) 
    model_s = set_model(_model_name_student, len(_labels_name), 
                            pretrained=_pretrained, neurons_reducer_block=_neurons_reducer_block)

    model_s = model_s.to(device)
    feat_fn_s = get_activation_fn(_model_name_student)

    #### get feat map dimension
    model_t.eval()
    model_s.eval()
    x_ = torch.randn(1,3,224, 224).to(device)
    out_s = feat_fn_s(model_s, x_, student=False)
    out_t = feat_fn_t(model_t, x_, student=False)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    regress_s = ConvReg(out_s[_layer_s].shape, out_t[_layer_t].shape)

    trainable_list.append(regress_s)

    models = {'model_s': model_s, 'model_t': model_t}
    feat_fn = {'feat_fn_s': feat_fn_s, 'feat_fn_t': feat_fn_t}
    _layers = {'_layer_s': _layer_s, '_layer_t': _layer_t}

    model_s.train()

    optimizer = optim.SGD(trainable_list.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience) 
    if _use_wce:
        wce_weight=torch.Tensor(_weights).to(device)
    else:
        wce_weight=None
    
    loss_fn = D_KD(weight=wce_weight, _layers=_layers, module_list=trainable_list, lambd=_lambd, 
                    lambd_rkd=_lambd_drkd, lambd_crkd=_lambd_crkd).to(device)

    ####################################################################################################################   
    print("- Starting the training phase...")
    print("-" * 50)

    fit_model_kd (models, train_data_loader, val_data_loader, optimizer=optimizer, feat_fn=feat_fn, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the test partition
    print("- Evaluating the test partition...")
    
    test_model_kd (models, test_data_loader, checkpoint_path=_checkpoint_best, feat_fn=feat_fn, loss_fn=loss_fn, save_pred=True,
                partition_name='test', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    
    torch.cuda.empty_cache()
    ####################################################################################################################

