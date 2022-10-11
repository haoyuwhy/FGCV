from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from typing import Union
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class FGVC_PIM(BaseModel):
    def __init__(self, opt):
        super(FGVC_PIM, self).__init__(opt)
        opt['network_g']["lables"]=opt["lables"]
        self.batchsize=opt["datasets"]["train"]["batch_size_per_gpu"]
        self.opt=opt
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path_backbone = self.opt['path'].get('pretrain_network_backbone', None)
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
             param_key = self.opt['path'].get('param_key_g', 'params_ema')
             self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        elif load_path_backbone is not None:
            param_key = self.opt['path'].get('param_key_backbone', 'params')
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                net = self.net_g.module
            self.load_network(net.net.backbone, load_path_backbone, self.opt['path'].get('strict_load_backbone', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path_backbone = self.opt['path'].get('pretrain_network_backbone', None)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_g', 'params_ema')
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            elif load_path_backbone is not None:
                param_key = self.opt['path'].get('param_key_backbone', 'params')
                self.load_network(self.net_g_ema.net.backbone, load_path_backbone, self.opt['path'].get('strict_load_backbone', True), param_key)
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('CrossentropyLoss_opt'):
            self.crossentropyLoss = build_loss(train_opt['CrossentropyLoss_opt']).to(self.device)
        else:
            self.crossentropyLoss = None

        self.setup_optimizers()
        self.setup_schedulers()




    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.input = data[0].to(self.device)
        self.lables= data[1].to(self.device)
        self.name=data[2]


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        outs = self.net_g(self.input)

        loss = 0
        loss_dict = OrderedDict()
        for name in outs:
            if "select_" in name:
                if not self.opt["use_selection"]:
                    raise ValueError("Selector not use here.")
                if self.opt["lambda_s"] != 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.opt["lables"]).contiguous()
                    loss_s = nn.CrossEntropyLoss()(logit, 
                                                    self.labels.unsqueeze(1).repeat(1, S).flatten(0))
                    loss += self.opt["lambda_s"] * loss_s
                else:
                    loss_s = 0

            elif "drop_" in name:
                if not self.opt["use_selection"]:
                    raise ValueError("Selector not use here.")
                if self.opt["lambda_n"]!= 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.opt["lables"]).contiguous()
                    n_preds = nn.Tanh()(logit)
                    labels_0 = torch.zeros([self.batchsize * S, self.opt["lables"]]) - 1
                    labels_0 = labels_0.to(self.device)
                    loss_n = nn.MSELoss()(n_preds, labels_0)
                    loss += self.opt["lambda_n"] * loss_n
                else:
                    loss_n = 0.0
                if "loss_n" in loss_dict:
                    loss_dict["loss_n"]+=loss_n
                else:
                    loss_dict["loss_n"]=loss_n

            elif "layer" in name:
                if not self.opt["use_fpn"]:
                    raise ValueError("FPN not use here.")
                if self.opt["lambda_b"] != 0:
                    ### here using 'layer1'~'layer4' is default setting, you can change to your own
                    loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), self.lables)
                    loss += self.opt["lambda_b"] * loss_b
                else:
                    loss_b = 0.0
                if "loss_b" in loss_dict:
                    loss_dict["loss_b"]+=loss_b
                else:
                    loss_dict["loss_b"]=loss_b
            
            elif "comb_outs" in name:
                if not self.opt["use_combiner"]:
                    raise ValueError("Combiner not use here.")

                if self.opt["lambda_c"]!= 0:
                    loss_c = nn.CrossEntropyLoss()(outs[name], self.lables)
                    loss += self.opt["lambda_c"] * loss_c
                if "loss_c" in loss_dict:
                    loss_dict["loss_c"]+=loss_c
                else:
                    loss_dict["loss_c"]=loss_c

            elif "ori_out" in name:
                loss_ori = F.cross_entropy(outs[name], self.labels)
                loss += loss_ori
                if "loss_ori" in loss_dict:
                    loss_dict["loss_ori"]+=loss_ori
                else:
                    loss_dict["loss_ori"]=loss_ori


        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.input)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.input)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        record["acc"] = dict(better='higher', val=float('-inf'), iter=-1)
        # record["loss"] = dict(better="lower", val=float('inf'), iter=-1)
        self.best_metric_results[dataset_name] = record

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        use_pbar = self.opt['val'].get('pbar', False)
        self.metric_results = {"acc": 0}
        self._initialize_best_metric_results(dataset_name)
        corrects = {}
        total_samples = {}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx}, Img {val_data[2][0]}')
            self.test()
            score_names = []
            scores = []
            
            if self.opt["use_fpn"]:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, self.output[this_name].mean(1), self.lables, this_name, scores, score_names)
            _average_top_k_result(corrects, total_samples, scores, self.lables,tops=[i for i in range(1, 5)])
            if self.opt["use_combiner"]:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, self.output["comb_outs"], self.lables, this_name, scores, score_names)
            # loss=self.crossentropyLoss(self.output, self.lables)
            # if self.output.argmax()==self.lables:
            #     self.metric_results["acc"]+=1
            # self.metric_results["loss"] += loss

        if use_pbar:
            pbar.close()

        best_top1 = 0.0
        self.best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    self.best_top1_name = name

        self.metric_results["acc"] = best_top1
        # self.metric_results["loss"] /= (idx + 1)
        # update the best metric result
        self._update_best_metric_result(dataset_name, "acc", self.metric_results["acc"], current_iter)
        # self._update_best_metric_result(dataset_name, "loss", self.metric_results["loss"], current_iter)
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter @ '+self.best_top1_name)
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects

@torch.no_grad()
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    
    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1, 3]) # return top-1, top-3, top-5 accuracy
    
    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor, tops: list = [1, 2, 3, 4, 5]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    
    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1