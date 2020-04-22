# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:05:25 2019

@ Original code provided by Jianzhu Ma

@ Modified by: Mark Lee
"""
import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import mean_squared_error
import math

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

def train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, feature_dim, model_save_folder, train_epochs, batch_size, learning_rate, loss_balance, permute):

	loss_map = {}
	for term, _ in term_size_map.items():
		loss_map[term] = nn.MSELoss()

        train_corr = []
        train_MSE = []
        train_RMSE = []
        test_corr = []
        test_MSE = []
        test_RMSE = []

        train_feature, train_label, test_feature, test_label, num_drugs = train_data

	model = ont.dcell_nn(term_size_map, term_direct_gene_map, dG, feature_dim, root, num_drugs)

        # Convert all tensors from Double to float to avoid Runtime Error
        train_feature = train_feature.float()
        train_label = train_label.float()
        test_feature = test_feature.float()
        test_label = test_label.float()

        if permute == 'Y':
            # Shuffle the train and test label for permutation test
            train_label=train_label[torch.randperm(train_label.size()[0])]
            test_label=test_label[torch.randperm(test_label.size()[0])]

	train_label_gpu = train_label.cuda()
	test_label_gpu = test_label.cuda()

	model.cuda()
	term_mask_map = create_term_mask(model.term_direct_gene_map, feature_dim)

	for name, param in model.named_parameters():
		term_name = name.split('_')[0]
		if '_direct_gene_layer.weight' in name:
			#print name, param.size(), term_mask_map[term_name].size()
			param.data = torch.mul(param.data, term_mask_map[term_name]) * loss_balance
		else:
			param.data = param.data * loss_balance

	train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
	test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

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

		train_accu = ut.average_spearman_corr(train_predict, train_label_gpu)
                train_corr.append(train_accu)
                train_MSE.append(mean_squared_error(train_predict.cpu(), train_label_gpu.cpu()))
                train_RMSE.append(math.sqrt(train_MSE[-1]))

		if epoch % 10 == 0:
			torch.save(model, model_save_folder + '/model_' + str(epoch))

		#Test
		model.eval()
		
		test_predict = torch.zeros(0,num_drugs).cuda()
		for i, (genotypes, labels) in enumerate(test_loader):
			# Convert torch tensor to Variable
			cuda_features = Variable(genotypes.cuda())

			aux_out_map,_ = model(cuda_features)
			test_predict = torch.cat([test_predict, aux_out_map[root].data],0)

		test_accu = ut.average_spearman_corr(test_predict, test_label_gpu)
                test_corr.append(test_accu)
                test_MSE.append(mean_squared_error(test_predict.cpu(), test_label_gpu.cpu()))
                test_RMSE.append(math.sqrt(test_MSE[-1]))


        # Create plots and csv file of train and test correlation, MSE and RMSE:
        ut.write_results_to_csv(train_corr, test_corr, train_MSE, test_MSE, train_RMSE, test_RMSE, model_save_folder + '/' + "results.csv")
        
        # Output results for each drug individually and save the predictions
        drugs = ['Doxorubicin','Tamoxifen','masitinib','17-AAG','GDC-0941','PLX4720']
        ut.output_stats_each_drug(test_predict.cpu(), test_label_gpu.cpu(), drugs, model_save_folder + '/drug_stats.csv')
        ut.save_results(test_predict.cpu(), drugs, model_save_folder + '/final_predictions.csv')
        ut.save_results(test_label_gpu.cpu(), drugs, model_save_folder + '/test_labels.csv')
            
	torch.save(model, model_save_folder + '/model_final')	
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
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=500)
parser.add_argument('-model', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-loss', help='Parameter to balance the 2 loss functions', type=float, default=0.2)
parser.add_argument('-permute', help='Y if permute test is to be done, N otherwise', type=str, default='N')
opt = parser.parse_args()

torch.set_printoptions(precision=5)

print("**********Parameters: **************")
print("Ontology: " + opt.onto.split('/')[-1])
print("Training: " + opt.train.split('/')[-1])
print("Testing: " + opt.test.split('/')[-1])
print("Epochs: ", opt.epochs)
print("Learning rate: ", opt.lr)
print("Batch size: ", opt.batchsize)
print("Loss balance: ", opt.loss)
print("Model directory: " + opt.model.split('/')[-1])
print("Permute?: " + opt.permute)
print("************************************")

train_data, gene2id_mapping = ut.load_GDSC_CCLE_data(opt.train, opt.test, opt.data)

dG, root, term_size_map, term_direct_gene_map = ut.load_ontology(opt.onto, gene2id_mapping)

ut.save_gene2id_mapping(gene2id_mapping, root, opt.model)

train_dcell(root, term_size_map, term_direct_gene_map, dG, train_data, len(gene2id_mapping), opt.model, opt.epochs, opt.batchsize, opt.lr, opt.loss, opt.permute)

