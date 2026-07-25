import copy
import numpy as np
import random
import uncertainty_toolbox as uct
import learn2learn as l2l
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from kdbnet.dta import KIBA, DAVIS
from kdbnet.model import DTAModel
from kdbnet.metrics import evaluation_metrics
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
from kdbnet.utils import(
    Logger,
    Saver,
    EarlyStopping
)
torch.set_num_threads(1)

def _parallel_train_per_epoch(
    kwargs=None, test_loader=None,
    n_epochs=None, eval_freq=None, test_freq=None,
    monitoring_score='pearson',
    loss_fn=None, logger=None,
    test_after_train=True,
):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    stopper = kwargs['stopper']
    best_model_state_dict = kwargs['best_model_state_dict']
    if stopper.early_stop:
        return kwargs
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / step
        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )
        if test_freq is not None and epoch % test_freq == 0:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                )
        if stopper.early_stop:
            logger.info('Eearly stop at epoch {}'.format(epoch))
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    rets = dict(midx = midx, model = model)
    return rets

def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson', 'r2']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results

def _unpack_evidential_output(output):
    mu, v, alpha, beta = torch.split(output, output.shape[1]//4, dim=1)
    inverse_evidence = 1. / ((alpha - 1) * v)
    var = beta * inverse_evidence
    return mu, var, inverse_evidence

class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method='protein',
        split_frac=[0.7, 0.1, 0.2],
        support_size=5,
        prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=1, n_epochs=5, batch_size=1,
        lr=0.001,
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        pretrain_epochs=50,
        output_dir='../output', save_log=False
    ):
        self.support_size = support_size
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)
        self.pretrain_epochs = pretrain_epochs
        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        dataset_klass = {
            'kiba': KIBA,
            'davis': DAVIS,
        }[task]
        self.dataset = dataset_klass(
            split_method=split_method,
            split_frac=split_frac,
            seed=seed,
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self._task_data_df_split = None
        self._task_loader = None
        n_gpus = torch.cuda.device_count()
        if self.parallel and n_gpus < self.n_ensembles:
            self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
            f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
            )
        self.devices = [torch.device(f'cuda:{i % n_gpus}')
            for i in range(self.n_ensembles)]
        self.model_config = dict(
            prot_emb_dim=1280,
            prot_gcn_dims=prot_gcn_dims,
            prot_fc_dims=prot_fc_dims,
            drug_node_in_dim=[66, 1],
            drug_node_h_dims=drug_gcn_dims,
            drug_fc_dims=drug_fc_dims,
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout)
        self.inner_lr = 0.001
        self.outer_lr = 0.0001
        self.episode_num = 100
        self.task_num_per_episode = 2
        self.valid_task_num_per_episode = 2
        self.inner_steps = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择
        self.models = self.build_model()
        self.models[0] = self.models[0].to(self.device)
        if self.pretrain_epochs > 0:
            self.pretrain_encoders(self.pretrain_epochs)
        else:
            self.models = self.build_model()
            self.models[0] = self.models[0].to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.maml = l2l.algorithms.MAML(self.models[0], lr=self.inner_lr, first_order=False, allow_unused=True)
        # self.opt = optim.Adam(self.maml.parameters(), self.outer_lr)
        self.maml = None
        self.opt = None
        self.split_method = split_method
        self.split_frac = split_frac
        self.current_inner_lr = 0.001
        self.lr_increment = 0.00001
        self.logger.info(self.models[0])
        self.similarity_sequence = 0
        # self.fai_alpha = 0.002
        # self.logger.info(self.optimizers[0])

    def _compute_intra_similarity(self, prot_list):

        if len(prot_list) < 2:
            return 0.0
        similarities = []
        for i in range(len(prot_list)):
            for j in range(i + 1, len(prot_list)):
                try:
                    f1 = prot_list[i].node_s.mean(dim=0)
                    f2 = prot_list[j].node_s.mean(dim=0)
                    sim = torch.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                    similarities.append(sim)
                except:
                    pass
        return np.mean(similarities) if similarities else 0.0

    def compute_task_difficulty(self, S_y, S_xp):
        S_y_tensor = torch.as_tensor(S_y, dtype=torch.float32)

        variance = torch.var(S_y_tensor).item()

        paired = list(zip(S_y, S_xp))
        sorted_pairs = sorted(paired, key=lambda x: x[0])
        mid = len(sorted_pairs) // 2

        class1_prots = [p for _, p in sorted_pairs[:mid]]
        class2_prots = [p for _, p in sorted_pairs[mid:]]

        sim1 = self._compute_intra_similarity(class1_prots)
        sim2 = self._compute_intra_similarity(class2_prots)
        avg_sim = (sim1 + sim2) / 2
        phi = abs(sim1 - sim2)

        difficulty = variance + (1.0 - avg_sim) + (1.0 - phi)
        return difficulty, variance

    def pretrain_encoders(self, pretrain_epochs=50, pretrain_batch_size=64):
        from tqdm import tqdm
        from torch_geometric.loader import DataLoader

        print("=" * 50)
        print("Pre-training ENTIRE MODEL (all parameters)...")
        print("=" * 50)
        train_loader = DataLoader(
            self.dataset.train_set,
            batch_size=pretrain_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )

        pretrain_model = DTAModel(**self.model_config).to(self.device)

        optimizer = optim.Adam(pretrain_model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()

        total_batches = len(train_loader)
        print(f"Total batches per epoch: {total_batches}")
        print("Trainable: ALL parameters (drug_model + prot_model + top_fc + fai_* + ...)")

        for epoch in range(pretrain_epochs):
            total_loss = 0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{pretrain_epochs}")
            for batch in pbar:
                xd = batch['drug'].to(self.device)
                xp = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)

                optimizer.zero_grad()
                yh = pretrain_model(xd, xp)
                loss = criterion(yh, y.view(-1, 1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{pretrain_epochs} completed, Average Loss: {avg_loss:.4f}")

        pretrained_state = pretrain_model.state_dict()
        self.models = self.build_model()
        self.models[0].load_state_dict(pretrained_state)
        self.models[0] = self.models[0].to(self.device)

        torch.save(pretrained_state, 'pretrained_model.pth')
        print(f"Pre-trained weights saved to 'pretrained_model.pth'")
        print("=" * 50)
        print("Pre-training complete!")
        print("  - ALL parameters are pre-trained and NOT frozen")
        print("  - Will be used for meta-testing directly")
        print("=" * 50)

    def build_model(self):
        models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        return models

    def get_inner_lr(self):
        self.current_inner_lr += self.lr_increment
        return self.current_inner_lr

    def _get_data_loader(self, dataset, shuffle=False):
        return torch_geometric.loader.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                )

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (data, df) = self.dataset.get_split(return_df=True)
            self._task_data_df_split = (data, df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
            test_df=None, esb_yp=None, recalib_df=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if esb_yp is not None:
            if self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def fast_adapt(self, adaptation_data1, adaptation_data2, adaptation_labels,
                   evaluation_data1, evaluation_data2, evaluation_labels,
                   learner, loss, adaptation_steps, device):

        from torch_geometric.loader import DataLoader
        import torch

        BATCH_SIZE = 1

        if isinstance(adaptation_data1, list) and len(adaptation_data1) > 0:
            drug_loader = DataLoader(adaptation_data1, batch_size=BATCH_SIZE)
            protein_loader = DataLoader(adaptation_data2, batch_size=BATCH_SIZE)

            all_s = []
            for drug_batch, protein_batch in zip(drug_loader, protein_loader):
                drug_batch = drug_batch.to(device)
                protein_batch = protein_batch.to(device)
                s = learner(drug_batch, protein_batch)
                s = s.squeeze()
                if s.dim() == 0:
                    s = s.unsqueeze(0)
                all_s.append(s)

            if all_s:
                combined_s = torch.cat(all_s, dim=0)
                train_error = loss(combined_s, adaptation_labels.to(device))

                trainable_params = [p for p in learner.parameters() if p.requires_grad]
                if len(trainable_params) > 0:
                    gradients = torch.autograd.grad(
                        train_error,
                        trainable_params,
                        allow_unused=True,
                        retain_graph=True
                    )
                    with torch.no_grad():
                        for param, grad in zip(trainable_params, gradients):
                            if grad is not None:
                                param.data = param.data - learner.lr * grad

        if isinstance(evaluation_data1, list) and len(evaluation_data1) > 0:
            drug_loader = DataLoader(evaluation_data1, batch_size=BATCH_SIZE)
            protein_loader = DataLoader(evaluation_data2, batch_size=BATCH_SIZE)

            all_preds = []
            for drug_batch, protein_batch in zip(drug_loader, protein_loader):
                drug_batch = drug_batch.to(device)
                protein_batch = protein_batch.to(device)
                pred = learner(drug_batch, protein_batch)
                pred = pred.squeeze()
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                all_preds.append(pred)

            if all_preds:
                self.predictions = torch.cat(all_preds, dim=0)
                valid_error = loss(self.predictions, evaluation_labels.to(device))
                return valid_error

        adaptation_data1 = adaptation_data1.to(device)
        adaptation_data2 = adaptation_data2.to(device)
        adaptation_labels = adaptation_labels.to(device)
        evaluation_data1 = evaluation_data1.to(device)
        evaluation_data2 = evaluation_data2.to(device)
        evaluation_labels = evaluation_labels.to(device)

        for step in range(adaptation_steps):
            s = learner(adaptation_data1, adaptation_data2)
            s = s.squeeze(1)
            train_error = loss(s, adaptation_labels)

            trainable_params = [p for p in learner.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                gradients = torch.autograd.grad(
                    train_error,
                    trainable_params,
                    allow_unused=True,
                    retain_graph=(step < adaptation_steps - 1)
                )
                with torch.no_grad():
                    for param, grad in zip(trainable_params, gradients):
                        if grad is not None:
                            param.data = param.data - learner.lr * grad

        predictions = learner(evaluation_data1, evaluation_data2)
        self.predictions = predictions.squeeze()
        valid_error = loss(self.predictions, evaluation_labels)
        return valid_error

    def process_data(self, data):
        all_feature_sequence = []
        for item in data:
            node_s = item.node_s
            node_v = item.node_v
            edge_s = item.edge_s
            edge_v = item.edge_v
            node_v_flattened = node_v.view(node_v.size(0), -1)
            node_features_combined = torch.cat((node_s, node_v_flattened), dim=1)
            edge_v_flattened = edge_v.view(edge_v.size(0), -1)
            edge_features_combined = torch.cat((edge_s, edge_v_flattened), dim=1)
            node_features_mean = node_features_combined.mean(dim=0, keepdim=True)
            edge_features_mean = edge_features_combined.mean(dim=0, keepdim=True)
            protein_features = torch.cat((node_features_mean, edge_features_mean), dim=1)
            all_feature_sequence.append(protein_features)
        return all_feature_sequence

    def cal_all_tasks_classes(self, data):
        all_tasks_classes = []
        for index, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(data):
            S_y_floats = [float(y) for y in S_y]
            items = [S_y_floats, S_xd, S_xp, Q_xd, Q_xp, Q_y]
            paired_items = list(zip(*items))
            sorted_pairs = sorted(paired_items, key=lambda x: x[0])
            mid_index = len(sorted_pairs) // 2
            class1 = sorted_pairs[:mid_index]
            class2 = sorted_pairs[mid_index:]
            S_y_class1 = [item[0] for item in class1]
            S_y_class2 = [item[0] for item in class2]
            S_xd_class1 = [item[1] for item in class1]
            S_xd_class2 = [item[1] for item in class2]
            S_xp_class1 = [item[2] for item in class1]
            S_xp_class2 = [item[2] for item in class2]
            task_classes = [
                [S_xd_class1, S_xp_class1, S_y_class1],
                [S_xd_class2, S_xp_class2, S_y_class2]
            ]
            all_tasks_classes.append(task_classes)
        return all_tasks_classes

    def cal_similarity(self, data):
        average_similarity_sequence = []
        all_mean_vector = []
        for task_classes in data:
            task_similarity_sequence = []
            for class_group in task_classes:
                feature_sequence = self.process_data(class_group[1])
                stacked_tensors = torch.stack(feature_sequence, dim=0)
                for i in range(len(feature_sequence)):
                    for j in range(i + 1, len(feature_sequence)):
                        tensor_i = feature_sequence[i][0].unsqueeze(0)
                        tensor_j = feature_sequence[j][0].unsqueeze(0)
                        similarity = F.cosine_similarity(tensor_i, tensor_j, dim=1).squeeze()
                        task_similarity_sequence.append(similarity.item())
                if task_similarity_sequence:
                    average_similarity = torch.tensor(task_similarity_sequence).mean()
                    average_similarity_sequence.append(average_similarity.item())
            mean_vector = stacked_tensors.mean(dim=0)
            all_mean_vector.append(mean_vector)
        return average_similarity_sequence, all_mean_vector

    def cal_pai(self, data):
        differences = []
        for i in range(0, len(data) - 1, 2):
            difference = abs(data[i] - data[i + 1])
            differences.append(difference)
        return differences

    def cal_meta_train_error(self, task_batch, tasks, t_com, t_rel):
        meta_train_error = 0.0
        aux_loss_total = 0.0

        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):
            difficulty, variance = self.compute_task_difficulty(S_y, S_xp)
            difficulty_normalized = torch.sigmoid(torch.tensor(difficulty - 2.0, dtype=torch.float32)).to(self.device)
            difficulty_normalized = difficulty_normalized.view(1)  # 变成 [1]

            variance_tensor = torch.tensor([variance], dtype=torch.float32).to(self.device)
            variance_embed = self.maml.module.fai_variance(variance_tensor)

            t_concatenate = torch.cat((t_rel[i].to(self.device),
                                       t_com[i].to(self.device),
                                       variance_embed.squeeze(0)))
            s_raw = self.maml.module.fai_t(t_concatenate)
            s = (1.0 - s_raw) / 2.0
            s = torch.clamp(s, 0.05, 0.95)

            aux_loss = torch.nn.functional.mse_loss(s, difficulty_normalized)
            aux_loss_total += aux_loss

            alpha_tensor = torch.stack([p for p in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            self.maml.lr = alpha
            learner = self.maml.clone()

            S_y_tensor = torch.as_tensor(S_y, dtype=torch.float32)
            Q_y_tensor = torch.as_tensor(Q_y, dtype=torch.float32)

            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y_tensor,
                                               Q_xd, Q_xp, Q_y_tensor,
                                               learner, self.criterion,
                                               self.inner_steps, self.device)
            meta_train_error = evaluation_error + meta_train_error

        lambda_aux = 0.1
        return meta_train_error + lambda_aux * aux_loss_total

    def cal_meta_valid_error(self, task_batch, tasks, t_com, t_rel):
        meta_valid_error = 0.0
        aux_loss_total = 0.0
        all_Q_y = []
        all_predictions = []

        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):

            difficulty, variance = self.compute_task_difficulty(S_y, S_xp)
            difficulty_normalized = torch.sigmoid(torch.tensor(difficulty - 2.0, dtype=torch.float32)).to(self.device)
            difficulty_normalized = difficulty_normalized.view(1)  # 变成 [1]

            variance_tensor = torch.tensor([variance], dtype=torch.float32).to(self.device)
            variance_embed = self.maml.module.fai_variance(variance_tensor)

            t_concatenate = torch.cat((t_rel[i].to(self.device),
                                       t_com[i].to(self.device),
                                       variance_embed.squeeze(0)))
            s_raw = self.maml.module.fai_t(t_concatenate)
            s = (1.0 - s_raw) / 2.0
            s = torch.clamp(s, 0.05, 0.95)

            aux_loss = torch.nn.functional.mse_loss(s, difficulty_normalized)
            aux_loss_total += aux_loss

            alpha_tensor = torch.stack([p for p in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            self.maml.lr = alpha
            learner = self.maml.clone()

            S_y_tensor = torch.as_tensor(S_y, dtype=torch.float32)
            Q_y_tensor = torch.as_tensor(Q_y, dtype=torch.float32)

            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y_tensor,
                                               Q_xd, Q_xp, Q_y_tensor,
                                               learner, self.criterion,
                                               self.inner_steps, self.device)
            meta_valid_error = evaluation_error + meta_valid_error
            all_Q_y.append(Q_y_tensor.cpu().numpy())
            all_predictions.append(self.predictions.detach().cpu().numpy())

        all_Q_y = np.concatenate(all_Q_y, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr, pearsonr
        from lifelines.utils import concordance_index

        mse = mean_squared_error(all_Q_y, all_predictions)
        r2 = r2_score(all_Q_y, all_predictions)
        ci = concordance_index(all_Q_y, all_predictions)
        spearman_corr, _ = spearmanr(all_Q_y, all_predictions)
        pearson_corr, _ = pearsonr(all_Q_y, all_predictions)

        lambda_aux = 0.1
        return meta_valid_error + lambda_aux * aux_loss_total, ci, r2, spearman_corr, pearson_corr

    def cal_meta_test_error(self, task_batch, tasks, t_com, t_rel):
        all_Q_y = []
        all_predictions = []
        self.opt.zero_grad()
        meta_adapting_error = 0.0
        aux_loss_total = 0.0
        best_model_path = 'best_model.pth'
        self.maml.load_state_dict(torch.load(best_model_path, map_location=self.device))

        for i, (S_xd, S_xp, S_y, Q_xd, Q_xp, Q_y) in enumerate(tasks):
            difficulty, variance = self.compute_task_difficulty(S_y, S_xp)
            difficulty_normalized = torch.sigmoid(torch.tensor(difficulty - 2.0)).to(self.device)

            variance_tensor = torch.tensor([variance], dtype=torch.float32).to(self.device)
            variance_embed = self.maml.module.fai_variance(variance_tensor)

            t_concatenate = torch.cat((t_rel[i].to(self.device),
                                       t_com[i].to(self.device),
                                       variance_embed.squeeze(0)))
            s_raw = self.maml.module.fai_t(t_concatenate)
            s = (1.0 - s_raw) / 2.0
            s = torch.clamp(s, 0.05, 0.95)

            aux_loss = torch.nn.functional.mse_loss(s, difficulty_normalized)
            aux_loss_total += aux_loss

            alpha_tensor = torch.stack([p for p in self.maml.module.fai_alpha])
            alpha = torch.mul(s, alpha_tensor[i])
            self.maml.lr = alpha
            learner = self.maml.clone()

            S_y_tensor = torch.as_tensor(S_y, dtype=torch.float32)
            Q_y_tensor = torch.as_tensor(Q_y, dtype=torch.float32)

            evaluation_error = self.fast_adapt(S_xd, S_xp, S_y_tensor,
                                               Q_xd, Q_xp, Q_y_tensor,
                                               learner, self.criterion,
                                               self.inner_steps, self.device)
            meta_adapting_error = evaluation_error + meta_adapting_error
            all_Q_y.append(Q_y_tensor.cpu().numpy())
            all_predictions.append(self.predictions.detach().cpu().numpy())

        all_Q_y = np.concatenate(all_Q_y, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        if len(all_Q_y) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, [], []

        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr, pearsonr
        from lifelines.utils import concordance_index

        mse = mean_squared_error(all_Q_y, all_predictions)
        r2 = r2_score(all_Q_y, all_predictions)
        ci = concordance_index(all_Q_y, all_predictions)
        spearman_corr, _ = spearmanr(all_Q_y, all_predictions)
        pearson_corr, _ = pearsonr(all_Q_y, all_predictions)

        return mse, ci, r2, spearman_corr, pearson_corr, all_Q_y, all_predictions

    def cal_t_com(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(self.task_num_per_episode, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com


    def cal_t_com_valid(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(self.valid_task_num_per_episode, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com

    def cal_t_com_test(self, data):
        all_tasks_classes = self.cal_all_tasks_classes(data)
        average_similarity_sequence, mean_vector = self.cal_similarity(all_tasks_classes)
        phi_sequence = self.cal_pai(average_similarity_sequence)
        phi_tensor = torch.tensor(phi_sequence, dtype=torch.float).view(self.valid_task_num_per_episode, 1)
        mean_vector_tensor = torch.stack(mean_vector).squeeze(1)
        pi_v = torch.cat((mean_vector_tensor, phi_tensor), dim=1).to(self.device)
        t_com = self.maml.module.fai_c(pi_v)
        return t_com

    def cal_t_rel(self, t_com):
        gamma = 1
        globalpool_A = self.maml.module.globalpool_A
        c_ij = torch.randn((t_com.shape[0], globalpool_A.shape[0]), dtype=torch.float32).to(self.device)
        for i in range(t_com.shape[0]):
            for j in range(globalpool_A.shape[0]):
                distances_squared = (t_com[i].unsqueeze(0) - globalpool_A).pow(2).sum(dim=1)
                numerator = (1 + distances_squared / gamma) ** (-(gamma + 1) / 2)
                c_ij[i, :] = numerator / numerator.sum()
        t_rel = torch.matmul(c_ij, globalpool_A)
        return t_rel

    def maml_train(self):
        if self.maml is None:
            self.maml = l2l.algorithms.MAML(
                self.models[0],
                lr=self.inner_lr,
                first_order=False,
                allow_unused=True
            )

        if self.opt is None:
            trainable_params = [p for p in self.maml.parameters() if p.requires_grad]
            self.opt = optim.Adam(trainable_params, self.outer_lr)
        if self.pretrain_epochs > 0:
            for param in self.models[0].drug_model.parameters():
                param.requires_grad = False
            for param in self.models[0].prot_model.parameters():
                param.requires_grad = False
            print("=" * 50)
            print("Encoders are FROZEN (pre-trained weights retained).")

            self.maml = l2l.algorithms.MAML(
                self.models[0],
                lr=self.inner_lr,
                first_order=False,
                allow_unused=True
            )
            trainable_params = [p for p in self.maml.parameters() if p.requires_grad]
            self.opt = optim.Adam(trainable_params, self.outer_lr)
            print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
            print("=" * 50)

        model_path = 'best_model.pth'
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"{model_path} has been deleted.")
        else:
            print(f"{model_path} does not exist.")

        best_spearman = -1
        best_ci = 0

        for iteration in range(self.episode_num):
            self.opt.zero_grad()

            train_task_batch, train_batch, tasks = self.dataset.get_batch_train_tasks(
                num_tasks=self.task_num_per_episode,
                support_size=self.support_size
            )
            t_com = self.cal_t_com(train_batch)
            t_rel = self.cal_t_rel(t_com)
            meta_train_error = self.cal_meta_train_error(
                train_task_batch, train_batch, t_com, t_rel
            )

            print('Iteration: ', iteration, ' Meta Train Error: ',
                  meta_train_error.item() / self.task_num_per_episode)

            meta_train_error = meta_train_error / self.task_num_per_episode
            meta_train_error.backward()
            self.opt.step()

            valid_task_num = 10
            if iteration >= self.episode_num - 5:
                all_mse = 0
                all_ci = 0
                all_r2 = 0
                all_spearman = 0
                all_pearson = 0

                for _ in range(valid_task_num):
                    train_task_batch, valid_batch, tasks = self.dataset.get_batch_valid_tasks(
                        num_tasks=self.valid_task_num_per_episode,
                        support_size=self.support_size
                    )
                    t_com = self.cal_t_com_valid(valid_batch)
                    t_rel = self.cal_t_rel(t_com)
                    mse, ci, r2, spearman_corr, pearson_corr = self.cal_meta_valid_error(
                        train_task_batch, valid_batch, t_com, t_rel
                    )
                    all_mse += mse
                    all_ci += ci
                    all_r2 += r2
                    all_spearman += spearman_corr
                    all_pearson += pearson_corr

                avg_mse = all_mse / valid_task_num
                avg_ci = all_ci / valid_task_num
                avg_r2 = all_r2 / valid_task_num
                avg_spearman = all_spearman / valid_task_num
                avg_pearson = all_pearson / valid_task_num

                print('-------------Verification result---------------')
                print(f'MSE: {avg_mse:.4f}')
                print(f'CI: {avg_ci:.4f}')
                print(f'R²: {avg_r2:.4f}')
                print(f'Spearman: {avg_spearman:.4f}')
                print(f'Pearson: {avg_pearson:.4f}')

                learner = self.maml.clone()
                if avg_ci > best_ci:
                    best_ci = avg_ci
                    best_model_state_dict = learner.state_dict()
                    print(f'New best model with ci {best_ci:.4f} saved.')
                    torch.save(best_model_state_dict, 'best_model.pth')

    def maml_test(self):
        self.opt.zero_grad()
        best_model_path = 'best_model.pth'
        self.maml.load_state_dict(torch.load(best_model_path, map_location=self.device))

        print("\n" + "=" * 50)
        print("Testing: 5-shot on Davis (Random Task Split)")
        print("=" * 50)

        all_mse = 0
        all_ci = 0
        all_r2 = 0
        all_spearman = 0
        all_pearson = 0
        testing_task_num = 50
        all_Q_y = []
        all_predictions = []

        for _ in range(testing_task_num):
            task_test_batch, test_batch, tasks = self.dataset.get_batch_test_tasks(
                num_test_tasks=2,
                support_size=self.support_size
            )
            if len(test_batch) == 0:
                continue
            t_com = self.cal_t_com_test(test_batch)
            t_rel = self.cal_t_rel(t_com)
            mse, ci, r2, spearman_corr, pearson_corr, y_true, y_pred = self.cal_meta_test_error(
                task_test_batch, test_batch, t_com, t_rel
            )
            all_mse += mse
            all_ci += ci
            all_r2 += r2
            all_spearman += spearman_corr
            all_pearson += pearson_corr
            all_Q_y.extend(y_true)
            all_predictions.extend(y_pred)

        all_Q_y = np.array(all_Q_y)
        all_predictions = np.array(all_predictions)
        print("\n" + "-" * 30)
        print("Debug: Prediction vs True Value Range")
        print(f"True value range: {np.min(all_Q_y):.2f} - {np.max(all_Q_y):.2f}")
        print(f"Predicted value range: {np.min(all_predictions):.2f} - {np.max(all_predictions):.2f}")
        print(f"True value mean: {np.mean(all_Q_y):.2f}")
        print(f"Mean of predicted values: {np.mean(all_predictions):.2f}")

        ss_res = np.sum((all_Q_y - all_predictions) ** 2)
        ss_tot = np.sum((all_Q_y - np.mean(all_Q_y)) ** 2)
        r2_manual = 1 - ss_res / ss_tot
        print(f"Manual calculation R²: {r2_manual:.4f}")
        print("-" * 30 + "\n")

        print(f"MSE: {all_mse / testing_task_num:.4f}")
        print(f"CI: {all_ci / testing_task_num:.4f}")
        print(f"R²: {all_r2 / testing_task_num:.4f}")
        print(f"Spearman: {all_spearman / testing_task_num:.4f}")
        print(f"Pearson: {all_pearson / testing_task_num:.4f}")

    def train(self, n_epochs=None, patience=None,
              eval_freq=1, test_freq=None,
              monitoring_score='pearson',
              train_data=None, valid_data=None,
              rebuild_model=False,
              test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx=i + 1,
                model=self.models[i],
                optimizer=self.optimizers[i],
                device=self.devices[i],
                train_loader=tl,
                valid_loader=vl,
                stopper=stp,
                best_model_state_dict=None,
            )
            rets_list.append(rets)
        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']

    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)
        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }
        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'spearman', 'pearson', 'r2']
        )
        results['metrics'] = eval_metrics
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results

    def test_pretrained_only(self):

        print("=" * 50)
        print("Testing: 5-shot with PRE-TRAINED MODEL (no meta-training)")
        print("=" * 50)

        # 使用预训练好的模型
        if hasattr(self, 'models') and self.models[0] is not None:
            test_model = self.models[0]
            print("Loaded pre-trained model.")
        else:
            print("No pre-trained model found! Please run pre-training first.")
            return
        all_Q_y = []
        all_predictions = []
        testing_task_num = 24
        for _ in range(testing_task_num):
            task_proteins = list(set([d['protein_name'] for d in self.dataset.test_set]))
            if len(task_proteins) == 0:
                continue
            selected_protein = random.choice(task_proteins)
            drug_records = [rec for rec in self.dataset.test_set if rec['protein_name'] == selected_protein]

            if len(drug_records) <= self.support_size:
                support_records = drug_records
                query_records = []
            else:
                shuffled = drug_records.copy()
                random.shuffle(shuffled)
                support_records = shuffled[:self.support_size]
                query_records = shuffled[self.support_size:]

            if len(query_records) == 0:
                continue

            support_drug = [rec['drug'] for rec in support_records]
            support_protein = [rec['protein'] for rec in support_records]
            support_y = [rec['y'] for rec in support_records]

            support_loader = DataLoader(support_drug, batch_size=len(support_drug))
            support_protein_loader = DataLoader(support_protein, batch_size=len(support_protein))

            support_drug_batch = next(iter(support_loader)).to(self.device)
            support_protein_batch = next(iter(support_protein_loader)).to(self.device)
            support_y = torch.tensor(support_y).to(self.device)

            learner = copy.deepcopy(test_model)
            learner.train()

            for param in learner.drug_model.parameters():
                param.requires_grad = False
            for param in learner.prot_model.parameters():
                param.requires_grad = False

            optimizer = optim.Adam(learner.top_fc.parameters(), lr=0.001)

            for step in range(self.inner_steps):
                yh = learner(support_drug_batch, support_protein_batch)
                loss = self.criterion(yh, support_y.view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            query_drug = [rec['drug'] for rec in query_records]
            query_protein = [rec['protein'] for rec in query_records]
            query_y = [rec['y'] for rec in query_records]

            query_loader = DataLoader(query_drug, batch_size=len(query_drug))
            query_protein_loader = DataLoader(query_protein, batch_size=len(query_protein))

            query_drug_batch = next(iter(query_loader)).to(self.device)
            query_protein_batch = next(iter(query_protein_loader)).to(self.device)
            query_y = np.array(query_y)

            learner.eval()
            with torch.no_grad():
                yh = learner(query_drug_batch, query_protein_batch)
                y_pred = yh.squeeze().cpu().numpy()

                all_Q_y.extend(query_y)
                all_predictions.extend(y_pred)

        all_Q_y = np.array(all_Q_y)
        all_predictions = np.array(all_predictions)

        if len(all_Q_y) < 2:
            print("No test samples found!")
            return

        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import spearmanr, pearsonr
        from lifelines.utils import concordance_index

        mse = mean_squared_error(all_Q_y, all_predictions)
        r2 = r2_score(all_Q_y, all_predictions)
        ci = concordance_index(all_Q_y, all_predictions)
        spearman_corr = spearmanr(all_Q_y, all_predictions)[0]
        pearson_corr = pearsonr(all_Q_y, all_predictions)[0]

        print("\n" + "-" * 30)
        print("5-shot Results (Pre-trained Model, No Meta-Training):")
        print(f"  MSE: {mse:.4f}")
        print(f"  CI: {ci:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"  Pearson: {pearson_corr:.4f}")
        print("-" * 30)