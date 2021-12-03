import os
import yaml
import numpy as np
import time
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import matplotlib.pyplot as plt
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
## Progress bar
from tqdm.notebook import tqdm
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

import urllib.request
from urllib.error import HTTPError

# Create checkpoint path if it doesn't exist yet
DATASET_PATH = "./data"
CHECKPOINT_PATH = "./"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")

def get_config(filename):
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1,agg_type='add', **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          aggr = agg_type,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class MLPModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class NodeLevelGNN(pl.LightningModule):

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)

def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=200,
                         progress_bar_refresh_rate=0) # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    pl.seed_everything()
    model = NodeLevelGNN(model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc'],
              "number_params":sum(p.numel() for p in model.parameters() if p.requires_grad) }

    return model, result


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")
    print(f"Number parameters:  {(result_dict['number_params']):f}")



'''
node_mlp_model, node_mlp_result = train_node_classifier(model_name="MLP",
                                                        dataset=cora_dataset,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.1)

print_results(node_mlp_result)
'''
configs = get_config("config.yaml")

hidden_vector_sizes = [2,4,8,16,32,64]
num_layers = [1,2,4,6,8]
aggregation_functions = ['add', 'mean','max']
hidden_vector_sizes = [16]
num_layers = [2]
#aggregation_functions = ['add']
hidden_vector_size = 16
num_layer = 2
#testing_acc = []

#num_layers = [configs['gnn']['layers_message_passing']]
#aggregation_types = [configs['gnn']['aggregation_type']]
#hidden_vector_sizes = [configs['gnn']['embedding_size']]
max_epochs = configs['optim']['max_epoch']

testing_accuracies = []
number_parameters = []

start_time = time.time()
for hidden_vector_size in hidden_vector_sizes:
    for num_layer in num_layers:
        for aggregation_function in aggregation_functions :
            node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                                    layer_name="GCN",
                                                                    dataset=cora_dataset,
                                                                    c_hidden=hidden_vector_size,
                                                                    num_layers=num_layer,
                                                                    dp_rate=0.1,
                                                                    agg_type=aggregation_function)
            testing_accuracies.append(node_gnn_result['test'])
            number_parameters.append(node_gnn_result['number_params'])
            print_results(node_gnn_result)

def plot (x_value,y_value,x_axis_label, y_axis_label,categorical=False):
    #colors = np.random.rand(4,)
    colors = np.random.rand(len(y_value))
    if categorical == True:
        x_value = [str(elem) for elem in x_value]
        plt.bar(x_value, y_value)
    else :
        plt.xticks(x_value)
        plt.scatter(number_parameters, testing_accuracies, c=colors, alpha=0.5)
    plt.title(str(x_axis_label) + ' vs ' + str(y_axis_label))
    plt.xlabel(str(x_axis_label))
    plt.ylabel(str(y_axis_label))
    plt.savefig(str(x_axis_label) + '_vs_' + str(y_axis_label) + ".png")
    plt.show()

print ("Time taken to run" , time.time()-start_time)
#plot (hidden_vector_sizes,testing_accuracies,"Embedding size","Testing Accuracy",categorical=True)
#plot (num_layers,testing_accuracies,"Number of layers","Testing Accuracy",categorical=True)
plot (aggregation_functions,testing_accuracies,"Aggregation Function","Testing Accuracy",categorical=True)

