# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:05:25 2019

@ Original code provided by Jianzhu Ma

@ Modified by: Mark Lee
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import mean_squared_error
import math                                              
import random
import sklearn.preprocessing as sk
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.insert(0, 'jisoo_new/')
import util as ut
import ontology_NN as ont

def expand_genotype(genotype, feature_dim):
		
	feature = torch.zeros(genotype.size()[0], feature_dim).float()	
		
	for i in range(genotype.size()[0]):
		feature[i, genotype[i,0]] = 1
		feature[i, genotype[i,1]] = 1				

	return feature

def create_term_mask(term_direct_gene_map, feature_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), feature_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = mask.cuda()

		term_mask_map[term] = mask_gpu

	return term_mask_map

def train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, feature_dim, model_save_folder):

    train_feature, train_label, test_feature, test_label, num_drugs = train_data
    train_feature = train_feature.float()
    train_label = train_label.float()

    # Create model

    model = ont.dcell_nn(term_size_map, term_direct_gene_map, dG, feature_dim, root, num_drugs)
    model.cuda()
    term_mask_map = create_term_mask(model.term_direct_gene_map, feature_dim)
    
    ## Hyper-parameters settings
    #skf = KFold(n_splits=10, random_state=42)

    # Original settings:
    #ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001]
    #ls_batch = [16, 32, 64, 128, 256, 300]
    #ls_lb = [0.1, 0.15, 0.2, 0.25, 0.3] # loss balance term
    #ls_epochs = list(range(15, 51, 5))

    # Settings to try:
    ls_lr = [0.001]
    ls_batch = [32]
    ls_lb = [0.20, 0.25, 0.30] # loss balance term
    ls_epochs = [40]

    # For testing only- comment out later
    #skf = KFold(n_splits=2, random_state=42)
    #ls_lr = [0.01]
    #ls_batch = [256]
    #ls_lb = [0.2] # loss balance term
    #ls_epochs = [2]

    num_combinations = len(ls_lr) * len(ls_batch) * len(ls_lb) * len(ls_epochs)

    max_settings = min(100, num_combinations)

    print("ls_lr: ", ls_lr)
    print("ls_batch: ", ls_batch)
    print("ls_lb: ", ls_lb)
    print("ls_epochs: ", ls_epochs)


    used = list() # store previously used settings
    
    for setting in range(max_settings):
        
        while True:
            learning_rate = random.choice(ls_lr)
            batch_size = random.choice(ls_batch)
            loss_balance = random.choice(ls_lb)
            train_epochs = random.choice(ls_epochs)
            
            if [learning_rate, batch_size, loss_balance, train_epochs] not in used:
                used.append([learning_rate, batch_size, loss_balance, train_epochs])
                break

        print("****Setting: ****")
        print("Epochs: ", train_epochs)
        print("Learning rate: ", learning_rate)
        print("Batch size: ", batch_size)
        print("Loss balance: ", loss_balance)
        print("*****************")
       
        num_folds = 1

        for name, param in model.named_parameters():
            term_name = name.split('_')[0]
            if '_direct_gene_layer.weight' in name:
                #print name, param.size(), term_mask_map[term_name].size()
                param.data = torch.mul(param.data, term_mask_map[term_name]) * loss_balance
            else:
                param.data = param.data * loss_balance

        for train_index, test_index in skf.split(train_feature):
            print("** Fold: **", num_folds)
            train_set = train_feature[train_index,:]
            validation_set = train_feature[test_index,:]
            train_set_labels = train_label[train_index,:]
            validation_set_labels = train_label[test_index,:]

            train_labels_gpu = train_set_labels.cuda()
            validation_labels_gpu = validation_set_labels.cuda()
            
        
            loss_map = {}
            for term, _ in term_size_map.items():
                loss_map[term] = nn.MSELoss()
        
            training_MSE = []
            training_RMSE = []
            validation_MSE = []
            validation_RMSE = []
            train_corr = []
            validation_corr = []
        
            train_loader = du.DataLoader(du.TensorDataset(train_set,train_set_labels), batch_size=batch_size, shuffle=False)
            test_loader = du.DataLoader(du.TensorDataset(validation_set,validation_set_labels), batch_size=batch_size, shuffle=False)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
        
            for epoch in range(train_epochs):
       
                #Train
        	model.train();
        	train_predict = torch.zeros(0,num_drugs).cuda()
        
        	for i, (genotypes, labels) in enumerate(train_loader):
        	    # Convert torch tensor to Variable
        	    cuda_features = Variable(genotypes.cuda())
        	    cuda_labels = Variable(labels.cuda())
        
        	    # Forward + Backward + Optimize
        	    optimizer.zero_grad()  # zero the gradient buffer
        	    aux_out_map,_ = model(cuda_features);
        
        	    train_predict = torch.cat([train_predict, aux_out_map[root].data],0)
        
        	    total_loss = 0
        	    for term, loss in loss_map.items():
        	        outputs = aux_out_map[term]
        		if term == root:	
        		    total_loss += loss_map[term](outputs, cuda_labels)
        		else:
        		    total_loss += loss_balance * loss_map[term](outputs, cuda_labels)
        
        	    total_loss.backward()
        
        	    for name, param in model.named_parameters():
        	        if '_direct_gene_layer.weight' not in name:
        		    continue
        		term_name = name.split('_')[0]
        		#print name, param.grad.data.size(), term_mask_map[term_name].size()
        		param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
        
        	    optimizer.step()

        
                train_corr.append(ut.spearman_corr(train_predict, train_labels_gpu).item())
                training_MSE.append(mean_squared_error(train_predict.cpu(), train_set_labels))
                training_RMSE.append(math.sqrt(training_MSE[-1]))
        
        	#Test
        	model.eval()
        		
        	validation_predict = torch.zeros(0,num_drugs).cuda()
        	for i, (genotypes, labels) in enumerate(test_loader):
        	    # Convert torch tensor to Variable
        	    cuda_features = Variable(genotypes.cuda())
        
        	    aux_out_map,_ = model(cuda_features)
        	    validation_predict = torch.cat([validation_predict, aux_out_map[root].data],0)
        
        	validation_corr.append(ut.spearman_corr(validation_predict, validation_labels_gpu).item())
                validation_MSE.append(mean_squared_error(validation_predict.cpu(), validation_set_labels))
                validation_RMSE.append(math.sqrt(validation_MSE[-1]))
        
        	#if epoch % 10 == 0:
        	    #print('Epoch', epoch, 'train accuracy', train_accu, 'test accuracy', test_accu)
        
            plot_title = "mb_size = {}, lr = {}, epochs = {}, loss_balance = {}, fold {}".format(batch_size, learning_rate, train_epochs, loss_balance, num_folds)
            plot_fname = "mb_size_{}_lr_{}_epochs_{}_lb_{}_fold_{}".format(batch_size, learning_rate, train_epochs, loss_balance, num_folds)
            
            # Create plots of MSE and RMSE for training and validation:
            ut.create_plots(train_corr, validation_corr, "Correlation - " + plot_title, ["train", "validation"], model_save_folder + '/' + plot_fname + '_correl')
            ut.create_plots(training_MSE, validation_MSE, "MSE - " + plot_title, ["train", "validation"], model_save_folder + '/' + plot_fname + '_MSE')
            ut.create_plots(training_RMSE, validation_RMSE, "RMSE - " + plot_title, ["train", "validation"], model_save_folder + '/' + plot_fname + 'RMSE')
            ut.write_results_to_csv(train_corr, validation_corr, training_MSE, validation_MSE, training_RMSE, validation_RMSE, model_save_folder + '/' + plot_fname + ".csv")
            
            print("*Final results: *")
            print("train_corr: " + str(train_corr[-1]))
            print("validation_corr: " + str(validation_corr[-1]))
            print("train_MSE: " + str(training_MSE[-1]))
            print("validation_MSE: " + str(validation_MSE[-1]))
            print("train_RMSE: " + str(training_RMSE[-1]))
            print("validation_RMSE: " + str(validation_RMSE[-1]))

            num_folds += 1

            # Don't need to save model in parameter tuning stage
            #torch.save(model, model_save_folder + '/model_final')	
            #model.cpu()
            #model2 = torch.load(model_save_folder + '/model_final')
            #model2.eval()
        
            #cuda_features = Variable(test_feature.cuda())
            #aux_out_map,_ = model2(cuda_features); 
            #test_accu = ut.spearman_corr(aux_out_map[root].data, test_label_gpu)	
            #print('reload model accuracy', test_accu)

parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Full path of the training file', type=str)
parser.add_argument('-test', help='Full path of the testing file', type=str)
parser.add_argument('-data', help='Directory that contains all other GDSC and CCLE data required for parser', type=str)
#parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
#parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
#parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
#parser.add_argument('-batchsize', help='Batchsize', type=int, default=500)
parser.add_argument('-model', help='Folder for plots', type=str, default='PLOTS/')
#parser.add_argument('-loss', help='Parameter to balance the 2 loss functions', type=float, default=0.2)
opt = parser.parse_args()

torch.set_printoptions(precision=5)

print("**********Run parameters: **************")
print("Ontology: " + opt.onto.split('/')[-1])
print("Training: " + opt.train.split('/')[-1])
print("Testing: " + opt.test.split('/')[-1])
print("Model directory: " + opt.model.split('/')[-1])
print("************************************")

train_data, gene2id_mapping = ut.load_GDSC_CCLE_data(opt.train, opt.test, opt.data)

dG, root, term_size_map, term_direct_gene_map = ut.load_ontology(opt.onto, gene2id_mapping)

ut.save_gene2id_mapping(gene2id_mapping, root, opt.model)

train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, len(gene2id_mapping), opt.model)

