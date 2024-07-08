import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.set_default_dtype(torch.float64)



def train_model(model, config, dataset, optimizer, device):

    train_history = dict(total_loss=[], ae_loss=[], map_loss=[])
    val_history = dict(total_loss=[], ae_loss=[], map_loss=[])
    min_val_loss = np.Inf

    optimization_config = config["optimization"]
    dataset_config = config["dataset"]

    w_map = optimization_config["map_loss_weight"]

    total_batches_train, total_batches_val = dataset.get_batches_per_epoch()

    model.train()  # Set network model in training mode
    loop = tqdm(range(optimization_config["epochs"]))
    for epoch in loop:

        dataset.generate_epoch_snapshots() # Essential to change the dataset (shuffle, etc) from epoch to epoch

        loss_train_total = 0
        loss_train_ae_s_total = 0
        loss_train_map_total = 0

        # Training routine

        dataset.set_train_mode(True)
        dataset.reset_current_batch()
        for batch_number in range(total_batches_train):
            optimizer.zero_grad()
            snapshots_batch_pred, z, z_estimation = model(dataset)
            loss_train_ae_s_batch = F.mse_loss(snapshots_batch_pred, dataset.get_batch_snapshots(), reduction='mean')
            loss_train_map_batch = F.mse_loss(z_estimation, z, reduction='mean')
            loss_train_batch = loss_train_ae_s_batch + w_map * loss_train_map_batch
            loss_train_batch.backward()
            optimizer.step()
            loss_train_total += loss_train_batch.item()
            loss_train_ae_s_total += loss_train_ae_s_batch.item()
            loss_train_map_total += loss_train_map_batch.item()
            dataset.increase_current_batch()
        loss_train_total /= total_batches_train
        loss_train_ae_s_total /= total_batches_train
        loss_train_map_total /= total_batches_train
            
        # scheduler.step()

        train_history['total_loss'].append(loss_train_total)
        train_history['ae_loss'].append(loss_train_ae_s_total)
        train_history['map_loss'].append(loss_train_map_total)        

        # Validation routine

        dataset.set_train_mode(False)
        dataset.reset_current_batch()
        with torch.no_grad():
            model.eval() # Set network model in evaluation mode
                
            loss_val_total = 0
            loss_val_ae_s_total = 0
            loss_val_map_total = 0
            
            for batch_number in range(total_batches_val):
                snapshots_batch_pred, z, z_estimation = model(dataset)
                loss_val_ae_s_batch = F.mse_loss(snapshots_batch_pred, dataset.get_batch_snapshots(), reduction='mean')
                loss_val_map_batch = F.mse_loss(z_estimation, z, reduction='mean')
                loss_val_batch = loss_val_ae_s_batch + w_map * loss_val_map_batch
                loss_val_total += loss_val_batch.item()
                loss_val_ae_s_total += loss_val_ae_s_batch.item()
                loss_val_map_total += loss_val_map_batch.item()
                dataset.increase_current_batch()
            loss_val_total /= total_batches_val
            loss_val_ae_s_total /= total_batches_val
            loss_val_map_total /= total_batches_val

        val_history['total_loss'].append(loss_val_total)
        val_history['ae_loss'].append(loss_val_ae_s_total)
        val_history['map_loss'].append(loss_val_map_total)   

        loop.set_postfix({"Loss(training)": train_history['total_loss'][-1], "Loss(validation)": val_history['total_loss'][-1]})

        if loss_val_total < min_val_loss:
            min_val_loss = loss_val_total
            best_epoch = epoch
            torch.save(model.state_dict(), dataset_config["output_root_path"]+'best_model_weights.pt')

        np.save(dataset_config["output_root_path"]+'train_history.npy', train_history)
        np.save(dataset_config["output_root_path"]+'val_history.npy', val_history)

    # Save network at last epoch
    torch.save(model.state_dict(), dataset_config["output_root_path"]+'last_model_weights.pt')
    
    print("\nBest network at epoch: ", best_epoch)
    
   