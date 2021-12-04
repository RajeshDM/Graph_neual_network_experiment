# Project submission for The A-Team group
This directory contains the code for running experiments for our project

### Installation:
`pip install -r requirements.txt`

### Running :
`python A_Team_project.py <option>`

Options available :   
help        : To see all the available options  
config      : Run GNN with configuration from the config.yaml file  
embedding   : To compare how different embedding sizes affect performance  
layers      : To compare how different number of GNN layers  affect performance  
aggregation : To compare how different aggregation functions affect performance  
all         : Cpompare how different number of parameters affect performance  


### Configuration :
All the GNN configurations are stored in the config.yaml file. To change the hyperparameters, please change them in the config.yaml file.  

In the config file, 

embedding\_size controls the size of all the embeddings in the graph representation  
layers\_message\_passing controls the number of GNN layers  
aggregation\_type controls the aggregation type (options are add, mean and max)  

You can change any of the above parameters to see the change in behaviour of the GNN 

### Teaching module:
The teaching module for the project is in the teaching module folder. This is the slide deck that we have created as the teaching module for the submission

