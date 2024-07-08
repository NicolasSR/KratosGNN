import numpy as np
from pathlib import Path
from sklearn import preprocessing

import torch
torch.set_default_dtype(torch.float64)


class Dataset():
            
    def __init__(self, node_coordinates, edges_list, pseudocoordinates, snapshots_train, snapshots_val, mu_train, mu_val, batch_size_train, batch_size_val, shuffle, device):

        self.device = device
        self.node_coordinates = torch.from_numpy(node_coordinates).to(self.device)
        self.edges_list = torch.from_numpy(edges_list).to(self.device)
        self.pseudocoordinates = torch.from_numpy(pseudocoordinates).to(self.device)
        self.snapshots_train = snapshots_train
        self.snapshots_val = snapshots_val
        self.mu_train = mu_train
        self.mu_val = mu_val


        if batch_size_train is None:
            batch_size_train = self.snapshots_train.shape[0]
        if batch_size_val is None:
            batch_size_val = self.snapshots_val.shape[0]
        self.number_of_batches_train = int(np.ceil(self.snapshots_train.shape[0]/batch_size_train))
        self.number_of_batches_val = int(np.ceil(self.snapshots_val.shape[0]/batch_size_val))

        self.shuffle = shuffle

        self.current_batch = 0

        self.train_mode = True # If True, will print data for training; if False, for validation

    def get_node_coordinates(self):
        return self.node_coordinates
    
    def get_edges_list(self):
        return self.edges_list
    
    def get_pseudocoordinates(self):
        return self.pseudocoordinates
    
    def increase_current_batch(self):
        self.current_batch += 1

    def reset_current_batch(self):
        self.current_batch = 0

    def set_train_mode(self, mode):
        self.train_mode = mode
    
    def generate_epoch_snapshots(self):
        
        self.current_batch = 0

        if self.shuffle:
            shuffled_indices_train = np.random.shuffle(np.arange(self.snapshots_train.shape[0]))
            shuffled_indices_val = np.random.shuffle(np.arange(self.snapshots_val.shape[0]))

            self.epoch_shapshots_train = self.snapshots_train[shuffled_indices_train].copy()
            self.epoch_shapshots_val = self.snapshots_val[shuffled_indices_val].copy()
            self.epoch_mu_train = self.mu_train[shuffled_indices_train].copy()
            self.epoch_mu_val = self.mu_val[shuffled_indices_val].copy()
        
        else:
            self.epoch_shapshots_train = self.snapshots_train.copy()
            self.epoch_shapshots_val = self.snapshots_val.copy()
            self.epoch_mu_train = self.mu_train.copy()
            self.epoch_mu_val = self.mu_val.copy()

        self.epoch_shapshots_train = [torch.from_numpy(array).to(self.device) for array in np.array_split(self.epoch_shapshots_train, self.number_of_batches_train, axis=0)]
        self.epoch_shapshots_val = [torch.from_numpy(array).to(self.device) for array in np.array_split(self.epoch_shapshots_val, self.number_of_batches_val, axis=0)]
        self.epoch_mu_train = [torch.from_numpy(array).to(self.device) for array in np.array_split(self.epoch_mu_train, self.number_of_batches_train, axis=0)]
        self.epoch_mu_val = [torch.from_numpy(array).to(self.device) for array in np.array_split(self.epoch_mu_val, self.number_of_batches_val, axis=0)]
    
    def get_batch_snapshots(self):
        if self.train_mode:
            return self.epoch_shapshots_train[self.current_batch]
        else:
            return self.epoch_shapshots_val[self.current_batch]
    
    def get_batch_mu(self):
        if self.train_mode:
            return self.epoch_mu_train[self.current_batch]
        else:
            return self.epoch_mu_val[self.current_batch]
    
    def get_batches_per_epoch(self):
        self.generate_epoch_snapshots()
        return len(self.epoch_shapshots_train), len(self.epoch_shapshots_val)



class DatasetGenerator():

    def __init__(self, config):

        self.optimization_config = config["optimization"]
        self.dataset_config = config["dataset"]

        edges_list_path = Path(self.dataset_config['dataset_root']+self.dataset_config['edges_list_path'])
        node_coordinates_path = Path(self.dataset_config['dataset_root']+self.dataset_config['node_coordinates_path'])
        if (not edges_list_path.is_file()) or (not node_coordinates_path.is_file()):
            print('Parsing MDPA file')
            self.parse_mdpa()

        self.node_coordinates = np.load(node_coordinates_path)
        edges_list_half1 = np.load(edges_list_path)
        edges_list_half2 = edges_list_half1.copy()[:,[1,0]]
        self.edges_list = np.vstack([edges_list_half1, edges_list_half2]).T

        self.pseudocoordinates = self.generate_pseudocoordinates()

        mu_train_path = Path(self.dataset_config['dataset_root']+self.dataset_config['mu_train_path'])
        mu_train_raw = np.load(mu_train_path)
        mu_val_path = Path(self.dataset_config['dataset_root']+self.dataset_config['mu_val_path'])
        mu_val_raw = np.load(mu_val_path)

        mu_scaler = self.get_scaler_function(self.dataset_config['mu_scaler'], mu_train_raw)
        self.mu_train = mu_scaler.transform(mu_train_raw)
        self.mu_val = mu_scaler.transform(mu_val_raw)

        self.channel_names = self.dataset_config['channel_names']
        self.dof_info_matrix = np.load(Path(self.dataset_config['dataset_root']+self.dataset_config['dof_info_matrix_path']))
        # idxs_channels_to_dofs, idxs_dof_to_channels = self.get_snapshot_reshaping_maps()
        self.generate_snapshot_reshaping_maps()

        snapshots_train_path = Path(self.dataset_config['dataset_root']+self.dataset_config['snapshots_train_path'])
        snapshots_val_path = Path(self.dataset_config['dataset_root']+self.dataset_config['snapshots_val_path'])
        snapshots_train_raw = np.load(snapshots_train_path).T
        snapshots_val_raw = np.load(snapshots_val_path).T

        snapshot_scaler = self.get_scaler_function(self.dataset_config['snapshot_scaler'], snapshots_train_raw)
        snapshots_train_scaled = snapshot_scaler.transform(snapshots_train_raw)
        snapshots_val_scaled = snapshot_scaler.transform(snapshots_val_raw)

        self.snapshots_train = self.reshape_dofs_to_channels_pt(torch.from_numpy(snapshots_train_scaled)).detach().cpu().numpy()
        self.snapshots_val = self.reshape_dofs_to_channels_pt(torch.from_numpy(snapshots_val_scaled)).detach().cpu().numpy()
        # self.snapshots_train = self.reshape_matrix(snapshots_train_scaled, idxs_dof_to_channels, idxs_channels_to_dofs.shape)
        # self.snapshots_val = self.reshape_matrix(snapshots_val_scaled, idxs_dof_to_channels, idxs_channels_to_dofs.shape)

        if np.all(self.reshape_channels_to_dofs_pt(torch.from_numpy(self.snapshots_train)).detach().cpu().numpy() == snapshots_train_raw):
            raise Exception('Snapshot reshaping process is not correct')
        
        self.batch_size_train = self.optimization_config["batch_size_train"]
        self.batch_size_val = self.optimization_config["batch_size_val"]
        self.shuffle = self.optimization_config["shuffle"]

    def parse_mdpa(self):

        node_coordinates_text = []
        elements_list_text = []
        print_node_coordinate = False
        print_element = False

        mdpa_path = Path(self.dataset_config['dataset_root']+self.dataset_config['mdpa_path'])
        with mdpa_path.open() as mdpa_file:
            for line in mdpa_file:

                if print_node_coordinate and ("End Nodes" in line):
                    print_node_coordinate = False
                elif print_node_coordinate:
                    node_coordinates_text.append(line[:-2])

                if "Begin Nodes" in line:
                    print_node_coordinate = True

                if print_element and ("End Elements" in line):
                    print_element = False
                elif print_element:
                    elements_list_text.append(line[:-2])

                if "Begin Elements" in line:
                    print_element = True
        
        node_coordinates = np.genfromtxt(node_coordinates_text, dtype=np.float64)[:,[1,2]] # Hardcoded for 2D problems. Change if needed
        elements_list = np.genfromtxt(elements_list_text, dtype=int)[:,[2,3,4]] # Hardcoded for triangular elements. Change if needed.

        edges_list = np.vstack([elements_list[:,[0,1]], elements_list[:,[1,2]]])
        edges_list = np.unique(np.sort(edges_list, axis=1), axis=0)
        edges_list = np.vstack([edges_list,edges_list[:,[1,0]]])
        edges_list -= 1 #MDPA file indexes nodes starting from 1, so we must correct

        np.save(Path(self.dataset_config['dataset_root']+self.dataset_config['edges_list_path']), edges_list)
        np.save(Path(self.dataset_config['dataset_root']+self.dataset_config['node_coordinates_path']), node_coordinates)

    def generate_snapshot_reshaping_maps(self):
        # We assume we will have a table with each DOF's index, the node it relates to and the type of value it relates to (in numerical code)

        map_shape = (self.get_number_of_nodes(), self.get_number_of_nodes()*len(self.channel_names))
        self.channel_maps_list = []
        for i in range(len(self.channel_names)):
            indices = []
            for dof in self.dof_info_matrix:
                if dof[2] == i:
                    indices.append([dof[1], dof[0]])
            values = np.ones(len(indices))
            indices = np.array(indices).T
            self.channel_maps_list.append(torch.sparse_coo_tensor(indices, values, map_shape).to_sparse_csr())

        # idxs_dofs_to_channels = np.empty((self.get_number_of_nodes(), len(self.channel_names)))
        # idxs_channels_to_dofs = np.empty((self.dof_info_matrix.shape[0], 2))

        # # Check dimensions
        # if self.get_number_of_nodes()*len(self.channel_names) != self.dof_info_matrix.shape[0]:
        #     raise Exception("Number of DoFs in dof_info_matrix does not correspond to number_of_modes * number_of_channels")
        
        # for dof in self.dof_info_matrix:
        #     idxs_dofs_to_channels[dof[1],dof[2]] = dof[0]
        #     idxs_channels_to_dofs[dof[0]] = [dof[1],dof[2]]

        # idxs_dofs_to_channels = idxs_dofs_to_channels.astype(int)
        # idxs_channels_to_dofs = idxs_channels_to_dofs.astype(int)

    def reshape_dofs_to_channels_pt(self, input_tensor):
        channels_values_list = []
        for map_tensor in self.channel_maps_list:
            channels_values_list.append(torch.unsqueeze(torch.transpose(torch.sparse.mm(map_tensor, torch.transpose(input_tensor,0,1)),0,1), 2))
        reshaped_tensor = torch.cat(channels_values_list, 2)
        return reshaped_tensor
    
    def reshape_channels_to_dofs_pt(self, input_tensor):
        channels_values_list_aux = torch.split(input_tensor, 1, dim=2)
        channels_values_list = []
        for channel_values in channels_values_list_aux:
            channels_values_list.append(torch.squeeze(channel_values, dim=2))
        reshaped_tensor = torch.transpose(torch.sparse.mm(torch.transpose(self.channel_maps_list[0],0,1),torch.transpose(channels_values_list[0],0,1)),0,1)
        for i in range(len(self.channel_maps_list)-1):
            reshaped_tensor += torch.transpose(torch.sparse.mm(torch.transpose(self.channel_maps_list[i+1],0,1), torch.transpose(channels_values_list[i+1],0,1)),0,1)
        return reshaped_tensor

    
    # def reshape_matrix(self, input_matrix, idxs_map, output_shape):
    #     output_matrix = np.empty((input_matrix.shape[0],*output_shape))
    #     for i in range(output_matrix.shape[0]):
    #         for j in range(idxs_map.shape[0]):
    #             output_matrix[i,j] = input_matrix[i,idxs_map[j]]
    #     return output_matrix

    def generate_pseudocoordinates(self):
        pseudocoordinates_list = []
        for node_i, node_j in self.edges_list.T:
            pseudocoordinates_list.append([self.node_coordinates[node_j,0]-self.node_coordinates[node_i,0], self.node_coordinates[node_j,1]-self.node_coordinates[node_i,1]])
        return np.array(pseudocoordinates_list)

    def get_scaler_function(self, scaler_name, input_matrix):
        if scaler_name == "minmax":
            sc_fun = preprocessing.MinMaxScaler()
        elif scaler_name == "robust":
            sc_fun = preprocessing.RobustScaler()
        elif scaler_name == "standard":
            sc_fun = preprocessing.StandardScaler()
        else:
            raise Exception('No valid scaler name was selected')
        
        return sc_fun.fit(input_matrix)

    def get_number_of_nodes(self):
        return self.node_coordinates.shape[0]

    def get_input_channels(self):
        return self.snapshots_train.shape[2]
    
    def get_mu_size(self):
        return self.mu_train.shape[1]
    
    def get_pseudocoordinates_dims(self):
        return self.pseudocoordinates.shape[1]

    def get_dataset(self, device):
        return Dataset(self.node_coordinates, self.edges_list, self.pseudocoordinates, self.snapshots_train, self.snapshots_val, self.mu_train, self.mu_val, self.batch_size_train, self.batch_size_val, self.shuffle, device)


                
            









