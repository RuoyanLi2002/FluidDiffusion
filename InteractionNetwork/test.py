import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

n_objects = 10
obj_dim = 2  # x pos, y pos

n_relations = n_objects * (n_objects - 1)
rel_dim = 1

eff_dim = 100
hidden_obj_dim = 100
hidden_rel_dim = 100

alpha = 0.5

device = torch.device("cpu")

class RelationModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RelationModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size),
			nn.ReLU()
		)

	def forward(self, x):
		'''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
		return self.model(x)


class ObjectModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(ObjectModel, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		'''
        Args:
            x: [n_objects, input_size]
        Returns:
            [n_objects, output_size]

        Note: output_size = number of states we want to predict
        '''
		return self.model(x)

rm = RelationModel(obj_dim * 2 + rel_dim, hidden_rel_dim, eff_dim)
om = ObjectModel(obj_dim + eff_dim, hidden_obj_dim, 2)

class InteractionNetwork(nn.Module):
	def __init__(self, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel, dim_x=0):
		super(InteractionNetwork, self).__init__()
		self.rm = rm
		self.om = om  # x, y

	def m(self, obj, rr, rs, ra):
		"""
		The marshalling function;
		computes the matrix products ORr and ORs and concatenates them with Ra

		:param obj: object states
		:param rr: receiver relations
		:param rs: sender relations
		:param ra: relation info
		:return:
		"""
		orr = obj.t().mm(rr)   # (obj_dim, n_relations)
		ors = obj.t().mm(rs)   # (obj_dim, n_relations)
		return torch.cat([orr, ors, ra.t()])   # (obj_dim*2+rel_dim, n_relations)

	def forward(self, obj, rr, rs, ra, x=None):
		"""
		objects, sender_relations, receiver_relations, relation_info
		:param obj: (n_objects, obj_dim)
		:param rr: (n_objects, n_relations)
		:param rs: (n_objects, n_relations)
		:param ra: (n_relations, rel_dim)
		:param x: external forces, default to None
		:return:
		"""
		# objects, sender_relations, receiver_relations, relation_info

		# marshalling function
		b = self.m(obj, rr, rs, ra)   # shape of b = (obj_dim*2+rel_dim, n_relations)

		# relation module
		e = self.rm(b.t())   # shape of e = (n_relations, eff_dim)
		e = e.t()   # shape of e = (eff_dim, n_relations)

		# effect aggregator
		if x is None:
			a = torch.cat([obj.t(), e.mm(rr.t())])   # shape of a = (obj_dim+eff_dim, n_objects)
		else:
			a = torch.cat([obj.t(), x, e.mm(rr.t())])   # shape of a = (obj_dim+ext_dim+eff_dim, n_objects)

		# object module
		p = self.om(a.t())   # shape of p = (n_objects, 2)

		return p

def format_data(data):
    objs = data   # (n_objects, obj_dim)
    
    receiver_r = torch.zeros((n_objects, n_relations), dtype=torch.float32)
    sender_r = torch.zeros((n_objects, n_relations), dtype=torch.float32)
    
    count = 0   # used as idx of relations
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                if torch.linalg.norm(objs[i, : ] - objs[j, : ]) < alpha:
                    receiver_r[i, count] = 1.0
                    sender_r[j, count] = 1.0
                count += 1
                
    r_info = torch.zeros((n_relations, rel_dim))
    
    return objs, sender_r, receiver_r, r_info

class My_InteractionNetwork(nn.Module):
    def __init__(self, dim_obj, dim_rel, dim_eff, dim_hidden_obj, dim_hidden_rel, dim_x=0):
        super(My_InteractionNetwork, self).__init__()
        self.rm = rm
        self.om = om  # x, y
    
    def m(self, x, edge_index):
        sender_idx = edge_index[0]
        receiver_idx = edge_index[1]

        # Pairwise distances
        sender_pos = x[sender_idx]  # (num_edges, 2)
        receiver_pos = x[receiver_idx]  # (num_edges, 2)

        ors = torch.zeros((obj_dim, edge_index.shape[1]), dtype=x.dtype, device=x.device, requires_grad=True)
        orr = torch.zeros((obj_dim, edge_index.shape[1]), dtype=x.dtype, device=x.device, requires_grad=True)

        ors = torch_scatter.scatter_add(sender_pos.t(), torch.arange(edge_index.shape[1], device=x.device).expand_as(sender_pos.t()), dim=1)
        orr = torch_scatter.scatter_add(receiver_pos.t(), torch.arange(edge_index.shape[1], device=x.device).expand_as(receiver_pos.t()), dim=1)

        r_info = r_info = torch.zeros((edge_index.shape[1], rel_dim))

        result = torch.cat([ors, orr, r_info.t()], dim=0)  # (obj_dim*2 + rel_dim, num_valid_edges)

        return result
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # marshalling function
        b = self.m(x, edge_index)   # shape of b = (obj_dim*2+rel_dim, n_relations)
        
        # relation module
        e = self.rm(b.t())   # shape of e = (n_relations, eff_dim)
        e = e.t()   # shape of e = (eff_dim, n_relations)
        
        # effect aggregator
        agg_result = torch.zeros((e.size(0), x.size(0)), dtype=e.dtype, device=e.device)
        sender_indices = edge_index[0]
        agg_result = torch_scatter.scatter_add(e, sender_indices.expand_as(e), dim=1, out=agg_result)
        
        a = torch.cat([x.t(), agg_result])   # shape of a = (obj_dim+eff_dim, n_objects)
        
        # object module
        p = self.om(a.t())   # shape of p = (n_objects, 2)
        
        return p

def create_graph(particle_pos):
    num_particles = particle_pos.shape[0]
    
    dist_matrix = torch.cdist(particle_pos, particle_pos, p=2)
    adj_matrix = (dist_matrix < alpha) & (dist_matrix > 0)
    
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    x = torch.tensor(particle_pos, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=None)

interaction_network = InteractionNetwork(obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)
my_interaction_network = My_InteractionNetwork(obj_dim, rel_dim, eff_dim, hidden_obj_dim, hidden_rel_dim)

particle_pos1 = torch.randn(n_objects, obj_dim)
particle_pos2 = torch.randn(n_objects, obj_dim)

objs, sender_r, receiver_r, r_info = format_data(particle_pos1)
temp1 = interaction_network(objs, receiver_r, sender_r,  r_info)
print(f"---1---temp1: {temp1}")

objs, sender_r, receiver_r, r_info = format_data(particle_pos2)
temp1 = interaction_network(objs, receiver_r, sender_r,  r_info)
print(f"---2---temp1: {temp1}")

g1 = create_graph(particle_pos1)
g2 = create_graph(particle_pos2)

dataloader = DataLoader([g1, g2], batch_size=2, shuffle=False)

# print(f"particle_pos1: {particle_pos1}")
# print(f"particle_pos2: {particle_pos2}")
for d in dataloader:
    # print(f"d: {d.x}")
    temp2 = my_interaction_network(d)
    print(f"temp2: {temp2}")