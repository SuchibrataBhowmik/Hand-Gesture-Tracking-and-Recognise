from argparse import ArgumentParser
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import library.models as mdl
from library.datasets import MyDataset
import library.metrics as met
import library.optim as optimz
from library.train_utils import Params, RunningAverage, load_checkpoint, load_json_to_dict, save_checkpoint, save_dict_to_json
from library.losses import  BCELogit_Loss
from library.profiling import Timer


device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
print("Device : ",device)

def main(dataset_dir):
    exp_dir = 'experiment'
    
    params = Params(join(exp_dir,'parameters.json'))

    metadata_train_file = join(exp_dir, "metadata.train")
    metadata_val_file = join(exp_dir, "metadata.valid")
    
    setup_timer = Timer(convert=True)
    setup_timer.reset()
    
    # Get the correct model
    if params.model == 'BaselineEmbeddingNet':
        model = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), upscale=params.upscale, corr_map_size=33, stride=4)
    elif params.model == 'VGG11EmbeddingNet_5c':
        model = mdl.SiameseNet(mdl.VGG11EmbeddingNet_5c(), upscale=params.upscale, corr_map_size=33, stride=4)
    elif params.model == 'VGG16EmbeddingNet_8c':
        model = mdl.SiameseNet(mdl.VGG16EmbeddingNet_8c(), upscale=params.upscale, corr_map_size=33, stride=4)
    
    model = model.to(device)
    print("\nLoading datasets .....")

    train_set = MyDataset(dataset_dir, dataset_type='train', upscale_factor=model.upscale_factor,
                            max_frame_sep=params.max_frame_sep, pos_thr=params.pos_thr, neg_thr=params.neg_thr,
                            cxt_margin=params.context_margin, metadata_file=metadata_train_file, save_metadata=metadata_train_file )
    train_loader = DataLoader(train_set, batch_size=params.batch_size,
                            shuffle=True, pin_memory=True)
                        
    valid_set = MyDataset(dataset_dir, dataset_type='validation', upscale_factor=model.upscale_factor,
                            max_frame_sep=params.max_frame_sep, pos_thr=params.pos_thr, neg_thr=params.neg_thr,
                            cxt_margin=params.context_margin, metadata_file=metadata_val_file, save_metadata=metadata_val_file )
    valid_loader = DataLoader(valid_set, batch_size=params.batch_size,
                            shuffle=True, pin_memory=True)
    print("Total train images : {} \nTotal validation image : {}".format(len(train_set),len(valid_set)))

    metrics = met.METRICS
    metrics['center_error']['kwargs']['upscale_factor'] = model.upscale_factor
    
    parameters = filter(lambda p: p.requires_grad,model.parameters())
    optimizer = optimz.OPTIMIZERS[params.optim](parameters, **params.optim_kwargs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.lr_decay)

    print("\nStarting training for {} epochs .....".format(params.num_epochs))
    with Timer(convert=True) as t:
        train_and_evaluate(model, train_loader, valid_loader, optimizer,scheduler, metrics, params, exp_dir)

    print("Total time taken to train {}".format(t.elapsed))
    print("DONE !!!")
    
def train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, scheduler, metrics, params, exp_dir):
    best_valid_auc = 0
    train_met_list, valid_met_list = [],[]

    best_checkpoint_path = join(exp_dir, 'best.pth.tar')
    if exists(best_checkpoint_path):
        load_checkpoint(best_checkpoint_path, model)
        print("Restored parameters from {}".format(best_checkpoint_path))
    best_metrics_path = join(exp_dir, "best_metrics_values.json")
    if exists(best_metrics_path):
        best_valid_auc = load_json_to_dict(best_metrics_path)['AUC']
        print("Restored AUC from {}".format(best_metrics_path))

    for epoch in range(params.num_epochs):
        print("\nEpoch {}/{}".format(epoch + 1, params.num_epochs))

        train_metrics = train(model, train_dataloader, optimizer, metrics)
        scheduler.step()
        valid_metrics = evaluate(model, valid_dataloader, metrics)

        train_met_list.append(train_metrics)
        valid_met_list.append(valid_metrics)
        
        valid_auc = valid_metrics['AUC']
        is_best = valid_auc >= best_valid_auc
        save_checkpoint({'epoch':epoch+1,'state_dict':model.state_dict(),'optim_dict':optimizer.state_dict()},
                                      is_best=is_best, checkpoint=exp_dir)
        save_dict_to_json(valid_metrics, is_best=is_best, bestepoch=epoch+1, filepath=exp_dir)
        if is_best:
            best_valid_auc = valid_auc

    train_AUCs = [ x['AUC'] for x in train_met_list ]
    valid_AUCs = [ x['AUC'] for x in valid_met_list ]
    plt.plot(valid_AUCs, label='Validation AUC')
    plt.plot(train_AUCs, label='Training AUC')
    plt.legend(frameon=False)
    plt.savefig(exp_dir+'/AUC.jpeg')
    plt.close()
    train_center_err = [ x['center_error'] for x in train_met_list ]
    valid_center_err = [ x['center_error'] for x in valid_met_list ]
    plt.plot(valid_center_err, label='Validation center error')
    plt.plot(train_center_err, label='Training center error')
    plt.legend(frameon=False)
    plt.savefig(exp_dir+'/center_error.jpeg')
    plt.close()
    train_losses = [ x['loss'] for x in train_met_list ]
    valid_losses = [ x['loss'] for x in valid_met_list ]
    plt.plot(valid_losses, label='Validation loss')
    plt.plot(train_losses, label='Training loss')
    plt.legend(frameon=False)    
    plt.savefig(exp_dir+'/loss.jpeg')    
    plt.close()

def train(model, dataloader, optimizer, metrics):
    model.train()       # set model to training mode

    summ = []
    loss_avg = RunningAverage()
    with tqdm(total=len(dataloader)) as progbar:
        for i, sample in enumerate(dataloader):
            # move to GPU if available
            ref_img_batch = sample['ref_frame'].to(device)
            search_batch = sample['srch_frame'].to(device)
            labels_batch = sample['label'].to(device)
            # compute model output and loss
            output_batch = model(ref_img_batch, search_batch)

            loss = BCELogit_Loss(output_batch, labels_batch)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_batch = output_batch.detach().cpu().numpy()
            labels_batch = labels_batch.detach().cpu().numpy()

            summary_batch = {metric_name: metric_dict['fcn'](output_batch, labels_batch, **(metric_dict['kwargs']))
                             for metric_name, metric_dict in metrics.items()}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            loss_avg.update(loss.item())
            progbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            progbar.update()
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print("Training metrics : ", metrics_mean)
    return metrics_mean

@torch.no_grad()
def evaluate(model, dataloader, metrics):
    model.eval()

    summ = []
    loss_avg = RunningAverage()
    with tqdm(total=len(dataloader)) as progbar:
        for i, sample in enumerate (dataloader):
            # move to GPU if available
            ref_img_batch = sample['ref_frame'].to(device)
            search_batch = sample['srch_frame'].to(device)
            labels_batch = sample['label'].to(device)           
            # compute model output
            output_batch = model(ref_img_batch, search_batch)
            loss = BCELogit_Loss(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            
            # compute all metrics on this batch
            summary_batch = {metric_name: metric_dict['fcn'](output_batch, labels_batch, **(metric_dict['kwargs']))
                             for metric_name, metric_dict in metrics.items()}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            loss_avg.update(loss.item())
            progbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            progbar.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print("Evaluate metrics : ", metrics_mean)
    return metrics_mean

def argparser():
    parser = ArgumentParser(description="Training for hand gesture")
    parser.add_argument("-d","--dataset", type=str, default='hand_dataset',  help="Enter dataset directory")
    return parser.parse_args()

if __name__ == '__main__':
    arg = argparser()
    dataset_dir = arg.dataset
    if exists(dataset_dir):
        main(dataset_dir)
    else:print('Dataset does not exists')

    
    
    
    