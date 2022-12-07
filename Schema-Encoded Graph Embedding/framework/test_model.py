import torch
import utils
from model import GraphClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cuda' if torch.cuda.is_available() else 

(train_dataset,
         test_dataset,
         val_dataset,
         feat_dim,
         relations) = utils.load_dgl_data('streamspot', None, cv_fold=4, homo=False, bidirected=False, timestamp=True)
# 定义模型结构
net_clone = GraphClassifier(
            relations=relations,
            feat_dim=feat_dim,
            embed_dim=64,
            dim_a=16,
            agg_type='sum',
            dropout=0.2,
            activation='elu',
            pool='sum',
            total_latent_dim=None,
            inter_dim=None,
            final_dropout=None,
            sopool_type=None
            # timestamp=args.timestamp
        )
# 加载模型参数
net_clone.load_state_dict(torch.load('/root/autodl-tmp/root/autodl-tmp/PROV/rahmen_graph-main/saved_models/streamspot_none/streamspot_5cv_10ep_bi_result/checkpoint.pt')['model_state_dict'])
acc, p, r, f1 = net_clone.eval_model(test_dataset,
            batch_size=1,
            num_workers=4,
            device=device)
print(acc, p, r, f1)