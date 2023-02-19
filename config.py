
import torch
class Config(object):

 
    data_path = "data/"
    virs = "result"
    num_workers = 8  
    batch_size = 4
    max_epoch = 4000
    lr1 = 1e-4  #
    lr2 = 1e-4  
    beta1 = 0.5
    
    inputc = 2
    n_labels = 4

    gpu = True
    save_model = True
    tensorboard = True
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:1')
    
    model_run_name = "run_5_gan_ae_mask_ci_new-"
    save_path = model_run_name +'/imgs/'  
    model_save = model_run_name + "/model/"

    save_path_nd = model_run_name +'/imgsnd/'  
    model_save_nd = model_run_name + "/modelnd/"
    save_every = 1 
    save_model = 5

    netd_path = None
    netg_path = None



opt = Config()