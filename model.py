import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable

import math


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(
            num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(
            num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(
            e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img,
                              self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real,
                              self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img,
                              self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(
            num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded,
                        self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(
            num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(
            rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class ConvR(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvR, self).__init__()
        self.embedding_dim = 100
        self.filter_size = 5
        self.hidden_size = 6
        self.emb_dim1 = 10
        self.emb_dim2 = self.embedding_dim  // self.emb_dim1
        self.relation_dim = args.relation_dim

        self.emb_e = torch.nn.Embedding(num_entities, self.embedding_dim , padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.relation_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.embedding_dim)
        self.bn_rel = torch.nn.BatchNorm1d(self.embedding_dim * self.filter_size * self.filter_size)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(self.embedding_dim * 6 * 6, self.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        print(e1_embedded.shape)
        rel_embedded = self.emb_rel(rel)
        rel_embedded = rel_embedded.squeeze(1)

        rel_embedded=self.bn_rel(rel_embedded)
        e1 = self.bn0(e1_embedded)

        e1 = self.inp_drop(e1) 
        rel_embedded = self.inp_drop(rel_embedded)


        filters = rel_embedded.view(-1,self.embedding_dim, 1, self.filter_size, self.filter_size)

        x=torch.zeros([e1.shape[0],self.embedding_dim, self.hidden_size, self.hidden_size]).cuda()
        for i in range(e1.shape[0]):
            x[i] = F.conv2d(e1[i].unsqueeze(0), filters[i], stride=1, padding=0)

        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

# Add your own model here
class ConvTransE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvTransE, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.relation_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.channels = 32#100, 200, 300
        self.kernel_size = 3#1,3,5...
        self.init_emb_size = 100 #100, 200, 300

        self.conv1 =  torch.nn.Conv1d(2, self.channels, self.kernel_size, stride=1, padding= int(math.floor(self.kernel_size/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(self.init_emb_size)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(self.init_emb_size*self.channels,self.init_emb_size)
        #self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        #self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(self.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        #emb_initial = self.emb_e(X)
        #e1_embedded_all = self.bn_init(emb_initial)
        #e1_embedded = e1_embedded_all[e1]
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        pred = F.sigmoid(x)

        return pred
        
#TODO: Code Conv-TransR

# Add your own model here


class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(
            num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = torch.sigmoid(output)

        return prediction
