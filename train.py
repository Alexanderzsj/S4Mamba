import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import time
import torch
import random
import threading
import argparse
import numpy as np
from torchvision import models,transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss,resize
from utils.evaluation import *
from utils.HSICommonUtils import normlize3D, ImageStretching, beijing_time  
from utils.visual_loss import plot_loss, plot_acc
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict, pred_cls_map_dl
from utils.data_load_operate import get_class_name
from utils.data_load_operate import standardization
from calflops import calculate_flops
from thop import profile
import shutil
from options import args
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils.init import init_weights


torch.autograd.set_detect_anomaly(True)
time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(save_folder):
    """
    Save the model
    """
    source_file = f'./models/{net_name}'
    target_file = f'{save_folder}/{net_name}'
    if not os.path.exists(target_file):
        shutil.copytree(source_file, target_file)


def make_data(name, path):
    """
    Make data for training and testing

    :param name: dataset name 
    :param path: dataset path
    """

    data, gt = data_load_operate.load_data(name, path)
    if args.is_argument: 
        data = standardization(data)

    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    img = ImageStretching(data)

    class_count = max(np.unique(gt))

    args.channels = channels
    args.num_classes = class_count
    return img, data, gt, gt_reshape, height, width, channels, class_count


setup_seed(seed=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
record_computecost = args.record_computecost
seed_list = args.seed_list

num_list = [args.train_samples, args.val_samples]
ratio_list = [args.train_ratio, args.val_ratio]  # [train_ratio,val_ratio]
flag_list = [0, 1]  # ratio or num, # 1: fixed number for each class, 0: ratio for each class

tra_val = num_list if args.flag_num_ratio == 1 else ratio_list


max_epoch = args.max_epoch
learning_rate = args.lr
net_name = args.model

paras_dict = {'net_name':net_name,'dataset':args.data_set_name,'tra_val':tra_val, 'lr':learning_rate,'seed_list':seed_list}



def train_model(model, split_image, args):
    data_set_path = args.data_set_path
    data_set_name = args.data_set_name

    work_dir = './'

    exp_name = 'trained_models'
    net_folder = f'{net_name}, {beijing_time()}'
    save_folder = os.path.join(work_dir, exp_name, data_set_name, net_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    # 设置logger
    save_log_path = os.path.join(save_folder,'train_tr:{}_val:{}.log'.format(tra_val[0],tra_val[1]))
    logger = setup_logger(
        name=data_set_name,
        logfile=save_log_path,
        use_color=True,
        model_name=args.model
    )

    torch.cuda.empty_cache()
    logger.info(save_folder)

    img, data, gt, gt_reshape, height, width, channels, class_count = make_data(name=data_set_name, path=data_set_path)


    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx,curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'exp{}_seed{}'.format(str(exp_idx + 1), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_gt_folder = os.path.join(save_single_experiment_folder, 'gt')
        save_predict_folder = os.path.join(save_single_experiment_folder, 'predict')
        loss_img = os.path.join(save_single_experiment_folder, 'loss_img')
        if not os.path.exists(save_gt_folder):
            os.makedirs(save_gt_folder)
        if not os.path.exists(save_predict_folder):
            os.makedirs(save_predict_folder)
        if not os.path.exists(loss_img):
            os.makedirs(loss_img)

        save_model_folder = f'{save_single_experiment_folder}/pth'
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)

        save_weight_path = os.path.join(save_model_folder, "best_tr{}_val{}.pth".format(tra_val[0], tra_val[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(tra_val[0], tra_val[1]))
        predict_save_path = os.path.join(save_predict_folder,'pred_vis_tr{}_val{}.png'.format(tra_val[0], tra_val[1]))
        gt_save_path = os.path.join(save_gt_folder, 'gt_vis_tr{}_val{}.png'.format(tra_val[0], tra_val[1]))

        save_model(save_folder)


        train_data_index, val_data_index, test_data_index, all_data_index, train_class_counts, val_class_counts, test_class_counts = data_load_operate.sampling(ratio_list, num_list, gt_reshape, class_count, flag_list[args.flag_num_ratio])
        index = (train_data_index, val_data_index, test_data_index)


        if args.model_type==1 or args.model_type==2 or args.model_type==3:
            data_total_index = np.arange(data.shape[0] * data.shape[1]) 
            data_padded = data_load_operate.data_pad_zero(data, args.patch_size)
            height_patched, width_patched, channels = data_padded.shape
            train_iter, test_iter, val_iter = data_load_operate.generate_iter_1(data_padded, height, width, gt_reshape, 
                                                            index, args.patch_size, args.batch_size, args.model_type, args.model_3D_spa)

            # load data for the cls map of all the labed samples
            all_iter = data_load_operate.generate_iter_2(data_padded, height, width, gt_reshape, all_data_index,
                                                        args.patch_size, args.batch_size, args.model_type, args.model_3D_spa)
            
            # load data for the cls map of the total samples
            total_iter = data_load_operate.generate_iter_2(data_padded,height, width, gt_reshape, data_total_index, args.patch_size,
                                                        args.batch_size, args.model_type, args.model_3D_spa)
            sample_list1 = [all_iter, all_data_index]
            sample_list2 = [total_iter]


        # build Model
        logger.info("finish the process of dataset, begin building model")
        net = model
        logger.info(paras_dict)

        net.apply(init_weights)
        net.to(device)

        train_loss_list = [0]
        train_acc_list = [0]
        val_loss_list = [0]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        args.scheduler = f"step_size={args.step_size}, decay={args.gamma}"

        if args.model_type==4:
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        elif args.model_type==1 or args.model_type==2 or args.model_type==3:
            loss_func = torch.nn.CrossEntropyLoss()
        
        logger.info(optimizer)
        logger.info(args.scheduler)

        best_loss = 99999
        if record_computecost:
            net.eval()
            input_spa = torch.randn(1, args.patch_size * 2 + 1, args.patch_size * 2 + 1, args.channels).to(device) # For 2D model
            input_spe = torch.randn(1, args.channels).to(device) # Spectral data
            input_3D = torch.randn(1, 1, args.patch_size * 2 + 1, args.patch_size * 2 + 1, args.channels).to(device) # For 3D model

            if args.model_type == 1:
                if args.model_3D_spa == 1:
                    flops, para = profile(net, (input_3D,)) # 注意这里是 (input_3D,)，一个元组
                    para = f"{para / 1e3:.2f} K"
                    flops = f"{flops / 1e9:.2f} GFLOPS"

                if args.model_3D_spa == 0:
                    flops, para = profile(net, (input_spa,)) # 注意这里是 (input_3D,)，一个元组
                    para = f"{para / 1e3:.2f} K"
                    flops = f"{flops / 1e9:.2f} GFLOPS"
                
            elif args.model_type == 2:
                flops, macs1, para = calculate_flops(model=net, input_shape=tuple(input_spe.shape))

            elif args.model_type == 3:
                flops, para = profile(net, (input_spa, input_spe))
                para = f"{para / 1e3:.2f} K"
                flops = f"{flops / 1e9:.2f} GFLOPS"


            logger.info("param:{},flops:{}".format(para, flops))

        time1 = time.perf_counter()
        best_val_acc = 0


        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            # batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()

            net.train()
            if args.model_type==1 or args.model_type==2 or args.model_type==3: # data for single spatial net
                train_loss_sum, batch_counter = 0.0, 0
                for batch in train_iter:
                    if args.model_type==1:
                        X, y = batch
                        X = X.to(device)
                    elif args.model_type==2:
                        X, y = batch
                        X = X.to(device)
                    elif args.model_type==3:
                        X_spa, X_spe, y = batch
                        X_spa, X_spe = X_spa.to(device), X_spe.to(device)
                    
                    y = y.to(device)
                
                    y_pred = net(X_spa, X_spe) if args.model_type==3 else net(X)
                    ls=loss_func(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    batch_counter += 1
                    train_loss_sum += ls.cpu().item()
                
                scheduler.step()
                train_loss = train_loss_sum / batch_counter
                train_loss_list.append(train_loss)
                logger.info('Ex {}, Train epoch:{}, loss:{:.5f}, lr:{}'.format(exp_idx+1, epoch, train_loss, scheduler.get_lr()))


            torch.cuda.empty_cache()
            time2 = time.perf_counter()
            train_time = time2 - time1
            Train_Time_ALL.append(train_time)


            ############################################### evaluate stage ###################################################
            net.eval()
            with torch.no_grad():
                evaluator.reset()

                if args.model_type in [1, 2, 3]:
                    OA, val_loss = evaluate_OA(val_iter, net, loss_func, device, args.model_type)
                    val_loss_list.append(val_loss)
                    val_acc_list.append(OA)
                    logger.info('Evaluate epoch:{}, OA:{:5f}, loss:{:.5f}'.format(epoch, OA, val_loss))
                

                # save weight
                if OA>=best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    # torch.save(net,save_weight_path)
                    torch.save(net.state_dict(), save_weight_path)

                if (epoch+1)%50==0:
                    save_single_predict_path = os.path.join(save_predict_folder,'predict_{}.png'.format(str(epoch+1)))
                    save_single_gt_path = os.path.join(save_gt_folder,'gt.png')

                    save_epoch_weight_path = os.path.join(save_model_folder, "{}_trained{}.pth".format(args.model, str(epoch+1)))
                    torch.save(net.state_dict, save_epoch_weight_path)

                    if args.model_type==4:
                        vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)

                    if args.model_type in [1,2,3]:
                        predict_epoch_path = os.path.join(save_predict_folder, "{}_{}.png".format(args.model, str(epoch + 1)))
                        # pred_cls_map_dl(sample_list2, net, gt, predict_epoch_path, args.model_type, device)
                        print("Start to predict the cls map through a new thread...")
                        pred_thread = threading.Thread(target=pred_cls_map_dl, args=(sample_list2, net, gt, predict_epoch_path, args.model_type, device))
                        pred_thread.start()


        plot_loss(train_loss_list, val_loss_list, save_path=os.path.join(loss_img, 'loss_img.png'))
        plot_acc(val_acc_list, save_path=os.path.join(loss_img, 'acc_img.png'))
        torch.cuda.empty_cache()
            
        

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        
        pred_test = []
        load_weight_path = save_weight_path
        net.update_params = None
        # best_net = copy.deepcopy(net)
        best_net = model

        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            if args.model_type in [1, 2, 3]:
                val_loss_sum, batch_counter = 0.0, 0
                for batch in test_iter:
                    if args.model_type==1:
                        X, y = batch
                        X = X.to(device)
                    elif args.model_type==2:
                        X, y = batch
                        X = X.to(device)
                    elif args.model_type==3:
                        X_spa, X_spe, y = batch
                        X_spa, X_spe = X_spa.to(device), X_spe.to(device)
                    
                    y = y.to(device)

                    time1 = time.perf_counter()
                    y_pred = net(X_spa, X_spe) if args.model_type==3 else net(X)
                    time2 = time.perf_counter()

                    ls = loss_func(y_pred, y.long())
                    val_loss_sum += ls.cpu().item()
                    batch_counter += 1
                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

                
                val_loss = val_loss_sum / batch_counter
                val_loss_list.append(val_loss)

                y_gt = gt_reshape[test_data_index] - 1
                val_dict = evaluate_metrics(y_gt, pred_test)
                OA_test = val_dict['OA']
                val_acc_list.append(OA_test)

                mAcc_test = val_dict['MACC']
                Kappa_test = val_dict['Kappa']
                mIOU_test = val_dict['MIoU']
                IOU_test = val_dict['IoU']
                Acc_test = val_dict['ACC']
                logger.info('Test result: loss:{:5f}, OA:{}, MACC:{}, Kappa:{}, MIOU:{}, IOU:{}, ACC:{}'.format(val_loss, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test)) 


                pred_cls_map_dl(sample_list1, net, gt, gt_save_path, args.model_type, device)
                pred_cls_map_dl(sample_list2, net, gt, predict_save_path, args.model_type, device)

        
        test_time = time2 - time1
        Test_Time_ALL.append(test_time)

        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA: %s', list(OA_ALL))
    logger.info('List of AA: %s', list(AA_ALL))
    logger.info('List of KPP: %s', list(KPP_ALL))

    logger.info('OA=%.5f +- %.5f', round(np.mean(OA_ALL) * 100, 2), round(np.std(OA_ALL) * 100, 2))
    logger.info('AA=%.5f +- %.5f', round(np.mean(AA_ALL) * 100, 2), round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp=%.5f +- %.5f', round(np.mean(KPP_ALL) * 100, 2), round(np.std(KPP_ALL) * 100, 2))

    logger.info("ALL_CLASS_ACC FOR 10 TIMES: %s", EACH_ACC_ALL.tolist())

    mean_acc_per_class = np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2)
    std_acc_per_class = np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2)
    logger.info('Acc per class mean: %s', mean_acc_per_class.tolist()) # 打印均值数组
    logger.info('Acc per class std: %s', std_acc_per_class.tolist())   # 打印标准差数组

    logger.info("Average training time=%.5f +- %.5f", round(np.mean(Train_Time_ALL), 2), round(np.std(Train_Time_ALL), 3))
    logger.info("Average testing time=%.5f +- %.5f", round(np.mean(Test_Time_ALL) * 1000, 2), round(np.std(Test_Time_ALL) * 1000, 3))



    # ouput the message of the results
    mean_result_path = os.path.join(save_folder,'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n*******************************************************Paramater Settings*******************************************************' \
                  + '\nSeed List: ' + str(list(seed_list)) \
                  + '\nLearning rate: ' + str(learning_rate) \
                  + '\nStep size:' + str(args.step_size) + ', gamma: ' + str(args.gamma) \
                  + '\nMax epoch: ' + str(max_epoch) \
                  + '\nTrain Samples: ' + str(ratio_list[0] if args.flag_num_ratio == 0 else args.train_samples) \
                  + '\nVal Samples: ' + str(ratio_list[1] if args.flag_num_ratio == 0 else args.val_samples) \
                  + '\nModel: ' + str(args.model) \
                  + '\nClass count: ' + str(class_count) \
                  + '\nBatch size: ' + str(args.batch_size) \
                  + '\nBest OA: ' + str(OA_ALL.max()) + ', Experiment id:' + str(OA_ALL.argmax() + 1) \
                  + '\n\n*******************************************************Mean result of ' + str(len(seed_list)) + 'times runs *******************************************************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nEACH_ACC_ALL values:\n' + str(EACH_ACC_ALL)\
                  + '\n\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(
        np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(
        np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()


    del net