import sys
import argparse
import random
import torch
import os

from kdbnet.parsing import add_train_args
from kdbnet.experiment import DTAExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KDBNet experiment')
    add_train_args(parser)
    parser.add_argument('--load_pretrained', action='store_true', default=False,
                        help='Load pre-trained weights from pretrained_model.pth')
    args = parser.parse_args()

    exp = DTAExperiment(
        task=args.task,
        split_method=args.split_method,
        contact_cutoff=args.contact_cutoff,
        num_rbf=args.num_rbf,
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        drug_gcn_dims=args.drug_gcn_dims,
        drug_fc_dims=args.drug_fc_dims,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        uncertainty=args.uncertainty,
        parallel=args.parallel,
        output_dir=args.output_dir,
        save_log=args.save_log,
        support_size=args.support_size,
        pretrain_epochs=0,
    )

    if args.load_pretrained:
        print("Loading pre-trained weights from 'pretrained_model.pth'...")
        exp.models[0].load_state_dict(torch.load('pretrained_model.pth', map_location=exp.device))
        exp.models[0] = exp.models[0].to(exp.device)
        print("Loaded successfully!")
    elif args.pretrain_epochs > 0:
        exp.pretrain_encoders(args.pretrain_epochs)

    exp.maml_train()
    exp.maml_test()