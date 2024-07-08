import numpy as np

import torch
import torch_geometric.nn as gnn

torch.set_default_dtype(torch.float64)


class GNNFactory():

    def __init__(self, dataset_generator, config):
        self.dataset_generator = dataset_generator
        self.arch_config = config["architecture"]
        self.optimization_config = config["optimization"]

        latent_size = self.arch_config["latent_size"]
        conv_layers_list = self.arch_config["conv_layers_list"]
        full_layer_size = self.arch_config["full_layer_size"]
        map_layers_list = self.arch_config["map_layers_size"]

        self.skip = self.arch_config["skip"]

        number_of_nodes = self.dataset_generator.get_number_of_nodes()
        snapshot_channels = self.dataset_generator.get_input_channels()
        mu_size = self.dataset_generator.get_mu_size()
        pseudocoords_dims = self.dataset_generator.get_pseudocoordinates_dims()

        # Define layers for the encoder
        self.encoder_conv_layers = torch.nn.ModuleList()
        for i, layer in enumerate(conv_layers_list):
            if i == 0:
                input_channels = snapshot_channels
            else:
                input_channels = conv_layers_list[i-1][0]
            self.encoder_conv_layers.append(gnn.GMMConv(input_channels, layer[0], dim=pseudocoords_dims, kernel_size=layer[1]))

        self.unflattened_shape = (number_of_nodes, conv_layers_list[-1][0])
        self.flattenned_shape = conv_layers_list[-1][0]*number_of_nodes

        self.encoder_dense_layers = torch.nn.ModuleList()
        self.encoder_dense_layers.append(torch.nn.Linear(self.flattenned_shape,full_layer_size))
        self.encoder_dense_layers.append(torch.nn.Linear(full_layer_size,latent_size))

        # Define layers for the decoder
        self.decoder_dense_layers = torch.nn.ModuleList()
        self.decoder_dense_layers.append(torch.nn.Linear(latent_size,full_layer_size))
        self.decoder_dense_layers.append(torch.nn.Linear(full_layer_size,self.flattenned_shape))
    
        self.decoder_conv_layers = torch.nn.ModuleList()
        for i in range(len(conv_layers_list)):
            j = len(conv_layers_list) - i - 1
            if i == 0:
                output_channels = snapshot_channels
            else:
                output_channels = conv_layers_list[i-1][0]
            self.decoder_conv_layers.append(gnn.GMMConv(conv_layers_list[i][0], output_channels, dim=pseudocoords_dims, kernel_size=conv_layers_list[i][1]))

        # Define layers for the map network
        self.map_layers = torch.nn.ModuleList()
        
        self.map_layers.append(torch.nn.Linear(mu_size,map_layers_list[0]))
        for i in range(len(map_layers_list)-1):
            self.map_layers.append(torch.nn.Linear(map_layers_list[i],map_layers_list[i+1]))
        self.map_layers.append(torch.nn.Linear(map_layers_list[-1],latent_size))
    
    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device used: ", device)
        return device

    def get_encoder_class(self):

        encoder_dense_layers = self.encoder_dense_layers
        encoder_conv_layers = self.encoder_conv_layers
        skip = self.skip
        flattenned_shape = self.flattenned_shape

        class GNN_Encoder(torch.nn.Module):

            def __init__(self):
                super().__init__()

                self.encoder_dense_layers = encoder_dense_layers
                self.encoder_conv_layers = encoder_conv_layers

            def forward(self, data):
                out = data.get_batch_snapshots()

                for layer in self.encoder_conv_layers:
                    out_conv = torch.nn.SiLU(layer(out, data.get_edges_list(), data.get_pseudocoordinates()))
                    if skip:
                        out += out_conv
                    else:
                        out = out_conv

                out = out.reshape(out.shape[0], flattenned_shape)
                for layer in self.encoder_dense_layers[:-1]:
                    out = torch.nn.SiLU(layer(out))
                out = self.encoder_dense_layers[-1](out)

                return out
            
        return GNN_Encoder
    
    def get_decoder_class(self):

        decoder_dense_layers = self.decoder_dense_layers
        decoder_conv_layers = self.decoder_conv_layers
        skip = self.skip
        unflattened_shape = self.unflattened_shape


        class GNN_Decoder(torch.nn.Module):

            def __init__(self):
                super().__init__()

                self.decoder_dense_layers = decoder_dense_layers
                self.decoder_conv_layers = decoder_conv_layers

            def forward(self, input, data):
                out = input

                for layer in self.decoder_dense_layers:
                    out = torch.nn.SiLU(layer(out))

                out = out.reshape(out.shape[0], *unflattened_shape)

                for layer in self.decoder_conv_layers:
                    out_conv = torch.nn.SiLU(layer(out, data.get_edges_list(), data.get_pseudocoordinates()))
                    if skip:
                        out += out_conv
                    else:
                        out = out_conv

                return out
            
        return GNN_Decoder
    
    def get_map_class(self):

        map_layers = self.map_layers

        class GNN_Map(torch.nn.Module):

            def __init__(self):
                super().__init__()

                self.map_layers = map_layers

            def forward(self, data):
                out = data.get_batch_mu()

                for layer in self.map_layers[:-1]:
                    out = torch.nn.SiLU(layer(out))
                out = self.map_layers[-1](out)

                return out
            
        return GNN_Map
    
    def get_Autoencoder_Net(self):

        encoder_net = self.get_encoder_class()()
        decoder_net = self.get_decoder_class()()
        map_net = self.get_map_class()()

        class AutoEncoderNet(torch.nn.Module):

            def __init__(self):
                super().__init__()

                self.encoder_net = encoder_net
                self.decoder_net = decoder_net
                self.map_net = map_net

            def forward(self, data):
                z = self.encoder_net.forward(data)
                z_estimation = self.map_net.forward(data)
                x = self.decoder_net.forward(z, data)
                return x, z, z_estimation
            
        return AutoEncoderNet()
            
    
    def get_Estimator_Net(self):

        decoder_net = self.get_decoder_class()()
        map_net = self.get_map_class()()

        class EstimatorNet(torch.nn.Module):

            def __init__(self):
                super().__init__()

                self.decoder_net = decoder_net
                self.map_net = map_net

            def forward(self, data):
                z_estimation = self.map_net.forward(data)
                x = self.decoder_net.forward(z_estimation, data)
                return x, z_estimation
            
        return EstimatorNet()
    
    def get_optimizer(self, network):
        lr = self.optimization_config["learning_rate"]
        weight_decay = self.optimization_config["weight_decay"]
        return torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)