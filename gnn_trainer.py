from ArchitectureFactories.gnn_factory import GNNFactory
from DatasetGenerators.dataset_generator import DatasetGenerator
from OptimizationStrategies.sloss_only_optimiztion import train_model

import torch
torch.set_default_dtype(torch.float64)

if __name__=="__main__":
    
    config = {
        "architecture": {
            "latent_size": 30,
            "conv_layers_list": [[10,10],[50,50],[50,50]],
            "full_layer_size": 800,
            "map_layers_size": [10,20],
            "skip": True
        },
        "optimization": {
            "epochs": 5000,
            "batch_size_train": 8,
            "batch_size_val": None,
            "shuffle": True,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "map_loss_weight": 0.1
        },
        "dataset": {
            "dataset_root": 'dataset_cantilever/',
            "node_coordinates_path": 'node_coordinates.npy',
            "edges_list_path": 'edges_list.npy',
            "mu_train_path": 'SnapshotsMatrices/mu_train.npy',
            "mu_val_path": 'SnapshotsMatrices/mu_val.npy',
            "mu_scaler": 'standard',
            "snapshots_train_path": 'SnapshotsMatrices/fom_snapshots.npy',
            "snapshots_val_path": 'SnapshotsMatrices/fom_snapshots_val.npy',
            "snapshot_scaler": 'standard',
            "dof_info_matrix_path": 'dof_info_matrix.npy',
            "channel_names": ['DISPLACEMENT_X', 'DISPLACEMENT_Y'],
            "mdpa_path": 'rubber_hyperelastic_cantilever.mdpa',
            "output_root_path": 'saved_model/'
        }
    }

    dataset_generator = DatasetGenerator(config)

    model_factory = GNNFactory(dataset_generator, config)

    device = model_factory.get_device()
    network = model_factory.get_Autoencoder_Net()
    network.to(device)
    print('Total number of trainable parameters: ', sum(p.numel() for p in network.parameters() if p.requires_grad))
    optimizer = model_factory.get_optimizer(network)

    dataset = dataset_generator.get_dataset(device)

    train_model(network, config, dataset, optimizer, device)
