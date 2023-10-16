import argparse
import random

from src.lap_solvers.hungarian import *
from src.load_data import *
from src.loss_fun import *

from utils.config import cfg

from model.dgcnn import *
from model.hyperGCN import *
from model.affinity_layer import Affinity
from model.transformer import Transformer
from model.sinkhorn import sinkhorn_rpm



torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Hyper Gyral Hinge Matching',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--learning_rate', type=int, default=0.00003,
    help='Learning rate')

parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')

parser.add_argument(
    '--GT_train_path', type=str,
    default='../data/GT_L/',
    help='Path to the directory of Ground Truth mat.')
parser.add_argument(
    '--GT_eval_path', type=str,
    default='/data/hzb/project/Graph_matching/GyralHinge_dataset/ground_truth/GT_allpoint_0118_delete/GT_test/',
    help='Path to the directory of Ground Truth mat.')

parser.add_argument(
    '--feature_path', type=str,
    default='../data/L/',
    help='Path to the directory of training mat.')

parser.add_argument(
    '--model_path', type=str, default='/data/hzb/project/Graph_matching/GraphMatching_model/HHGMA/AB/',
    help='Path to the directory of Ground Truth mat.')

parser.add_argument(
    '--epoch', type=int, default=4,
    help='Number of epoches')

parser.add_argument(
    '--out_matrix_path', type=str, default='/data/hzb/project/Graph_matching/HyperGmatcing_savematrix/',
    help='Path to the directory of Ground Truth mat.')





def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True





class HyperGHMatching(nn.Module):

    def __init__(self):
        super().__init__()

        self.permutationLoss = PermutationLoss()

        self.pointfeaturer1 = DGCNN(cfg.HHGM.FEATURES, 1, cfg.HHGM.FEATURE_EDGE_CHANNEL)

        self.matching_layer = 3
        self.conv1_1 = nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1)
        self.conv1_2 = nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1)
        self.conv1_3 = nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1)

        for i in range(self.matching_layer):
            if i == 0:
                self_hyperedge_layer = Siamese_Econv(2048, cfg.HHGM.GNN_FEAT)
            else:
                self_hyperedge_layer = Siamese_Econv(cfg.HHGM.GNN_FEAT, cfg.HHGM.GNN_FEAT)
            self.add_module('self_enn_layer_{}'.format(i), self_hyperedge_layer)

            self.add_module('affinity_s1_{}'.format(i), Affinity(cfg.HHGM.GNN_FEAT))
            self.add_module('affinity_s2_{}'.format(i), Affinity(cfg.HHGM.GNN_FEAT))
            self.add_module('affinity_s3_{}'.format(i), Affinity(cfg.HHGM.GNN_FEAT))
            self.add_module('InstNorm_layer_s1_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            self.add_module('InstNorm_layer_s2_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            self.add_module('InstNorm_layer_s3_{}'.format(i), nn.InstanceNorm2d(1, affine=True))

            self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.HHGM.GNN_FEAT * 2, cfg.HHGM.GNN_FEAT))
            self.add_module('hyperedge_to_node_layer_{}'.format(i),
                            Hyperedge_to_node_conv(cfg.HHGM.GNN_FEAT, cfg.HHGM.GNN_FEAT))
            if cfg.HHGM.USEATTEND == 'attentiontransformer':
                self.add_module('gmattend{}'.format(i), Transformer(2 * cfg.HHGM.FEATURE_EDGE_CHANNEL
                                                                    if i == 0 else cfg.HHGM.GNN_FEAT))

    def forward(self, data):
        node0, node1 = data['fixed_graph_node_feat'], data['warp_graph_node_feat']
        node0 = torch.tensor(node0, dtype=torch.float32).to(device)
        node1 = torch.tensor(node1, dtype=torch.float32).to(device)

        edge0, edge1 = data['fixed_hypergraph_matrix'], data['warp_hypergraph_matrix']
        edge0 = torch.tensor(edge0, dtype=torch.float32).to(device)
        edge1 = torch.tensor(edge1, dtype=torch.float32).to(device)

        src_brainarea, tgt_brainarea = data['fixed_brainarea_matrix'], data['warp_brainarea_matrix']


        fixed_hypergraph_edge_degree = torch.diag(torch.sum(edge0, dim=0))
        fixed_hypergraph_node_degree = torch.diag(torch.sum(edge0, dim=1))
        fixed_w = torch.diag(torch.ones(edge0.shape[1]))
        src_w = torch.tensor(fixed_w, dtype=torch.float32).to(device)
        src_e_deg_inv = torch.tensor(torch.inverse(fixed_hypergraph_edge_degree), dtype=torch.float32).to(device)
        src_n_deg_inv = torch.tensor(torch.inverse(fixed_hypergraph_node_degree), dtype=torch.float32).to(device)

        warp_hypergraph_edge_degree = torch.diag(torch.sum(edge1, dim=0))
        warp_hypergraph_node_degree = torch.diag(torch.sum(edge1, dim=1))
        warp_w = torch.diag(torch.ones(edge1.shape[1]))
        tgt_w = torch.tensor(warp_w, dtype=torch.float32).to(device)
        tgt_e_deg_inv = torch.tensor(torch.inverse(warp_hypergraph_edge_degree), dtype=torch.float32).to(device)
        tgt_n_deg_inv = torch.tensor(torch.inverse(warp_hypergraph_node_degree), dtype=torch.float32).to(device)


        gt = data['Ground_truth']
        gt = torch.reshape(gt, (1, gt.shape[0], gt.shape[1]), )
        gt = torch.tensor(gt, dtype=torch.float32).to(device)

        node0 = node0.T
        node0 = torch.reshape(node0, (1, node0.shape[0], node0.shape[1]), )
        node1 = node1.T
        node1 = torch.reshape(node1, (1, node1.shape[0], node1.shape[1]), )

        Node_src1, Edge_src1 = self.pointfeaturer1(node0.permute(0, 2, 1).contiguous())
        Node_tgt1, Edge_tgt1 = self.pointfeaturer1(node1.permute(0, 2, 1).contiguous())
        emb_src = Node_src1.permute(0, 2, 1).contiguous()
        emb_tgt = Node_tgt1.permute(0, 2, 1).contiguous()
        hyper_src = torch.reshape(edge0, (1, edge0.shape[0], edge0.shape[1]), )
        hyper_tgt = torch.reshape(edge1, (1, edge1.shape[0], edge1.shape[1]), )

        src_w = torch.reshape(src_w, (1, src_w.shape[0], src_w.shape[1]), )
        src_e_deg_inv = torch.reshape(src_e_deg_inv, (1, src_e_deg_inv.shape[0], src_e_deg_inv.shape[1]), )
        src_n_deg_inv = torch.reshape(src_n_deg_inv, (1, src_n_deg_inv.shape[0], src_n_deg_inv.shape[1]), )
        tgt_w = torch.reshape(tgt_w, (1, tgt_w.shape[0], tgt_w.shape[1]), )
        tgt_e_deg_inv = torch.reshape(tgt_e_deg_inv, (1, tgt_e_deg_inv.shape[0], tgt_e_deg_inv.shape[1]), )
        tgt_n_deg_inv = torch.reshape(tgt_n_deg_inv, (1, tgt_n_deg_inv.shape[0], tgt_n_deg_inv.shape[1]), )


        ###################
        for i in range(self.matching_layer):
            self_enn_layer = getattr(self, 'self_enn_layer_{}'.format(i))

            emb_edge_src, emb_edge_tgt = self_enn_layer(
                [torch.bmm(src_w, hyper_src.permute(0, 2, 1).contiguous()), emb_src],
                [torch.bmm(tgt_w, hyper_tgt.permute(0, 2, 1).contiguous()), emb_tgt])



            affinity_s1 = getattr(self, 'affinity_s1_{}'.format(i))
            affinity_s2 = getattr(self, 'affinity_s2_{}'.format(i))
            affinity_s3 = getattr(self, 'affinity_s3_{}'.format(i))


            s_s1 = affinity_s1(emb_edge_src[:, 0:5, :], emb_edge_tgt[:, 0:5, :])
            s_s2 = affinity_s2(emb_edge_src[:, 5:(hyper_src.shape[2] - hyper_src.shape[1]), :],
                               emb_edge_tgt[:, 5:(hyper_tgt.shape[2] - hyper_tgt.shape[1]), :])
            s_s3 = affinity_s3(emb_edge_src[:, (hyper_src.shape[2] - hyper_src.shape[1]):emb_edge_src.shape[1], :],
                               emb_edge_tgt[:, (hyper_tgt.shape[2] - hyper_tgt.shape[1]):emb_edge_tgt.shape[1], :])

            InstNorm_layer_s1 = getattr(self, 'InstNorm_layer_s1_{}'.format(i))
            InstNorm_layer_s2 = getattr(self, 'InstNorm_layer_s2_{}'.format(i))
            InstNorm_layer_s3 = getattr(self, 'InstNorm_layer_s3_{}'.format(i))

            s_s1 = InstNorm_layer_s1(s_s1[:, None, :, :]).squeeze(dim=1)
            s_s2 = InstNorm_layer_s2(s_s2[:, None, :, :]).squeeze(dim=1)
            s_s3 = InstNorm_layer_s3(s_s3[:, None, :, :]).squeeze(dim=1)

            log_s_s1 = sinkhorn_rpm(s_s1, n_iters=20, slack=cfg.HHGM.SKADDCR)
            log_s_s2 = sinkhorn_rpm(s_s2, n_iters=20, slack=cfg.HHGM.SKADDCR)
            log_s_s3 = sinkhorn_rpm(s_s3, n_iters=20, slack=cfg.HHGM.SKADDCR)

            s_s1 = torch.exp(log_s_s1)
            s_s2 = torch.exp(log_s_s2)
            s_s3 = torch.exp(log_s_s3)

            s = torch.zeros(1, emb_edge_src.shape[1], emb_edge_tgt.shape[1]).to(device)
            s[0, 0:5, 0:5] = s_s1
            s[0, 5:(hyper_src.shape[2] - hyper_src.shape[1]), 5:(hyper_tgt.shape[2] - hyper_tgt.shape[1])] = s_s2
            s[0, (hyper_src.shape[2] - hyper_src.shape[1]):emb_edge_src.shape[1],
            (hyper_tgt.shape[2] - hyper_tgt.shape[1]):emb_edge_tgt.shape[1]] = s_s3
            # plt.imshow(s[0].detach().cpu().numpy())

            cross_graph = getattr(self, 'cross_graph_{}'.format(i))
            emb_edge_src = cross_graph(torch.cat((emb_edge_src, torch.bmm(s, emb_edge_tgt)), dim=-1))
            emb_edge_tgt = cross_graph(torch.cat((emb_edge_tgt, torch.bmm(s.transpose(1, 2), emb_edge_src)), dim=-1))

            if cfg.HHGM.USEATTEND == 'attentiontransformer':
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_edge_src, emb_edge_tgt)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                src_w = torch.softmax(scores_src, dim=-1)
                tgt_w = torch.softmax(scores_tgt, dim=-1)


            hyper_src_matmul = torch.bmm(src_n_deg_inv, hyper_src)
            hyper_src_matmul = torch.bmm(hyper_src_matmul, src_w)
            hyper_src_matmul = torch.bmm(hyper_src_matmul, src_e_deg_inv)

            hyper_tgt_matmul = torch.bmm(tgt_n_deg_inv, hyper_tgt)
            hyper_tgt_matmul = torch.bmm(hyper_tgt_matmul, tgt_w)
            hyper_tgt_matmul = torch.bmm(hyper_tgt_matmul, tgt_e_deg_inv)

            hyperedge_to_node = getattr(self, 'hyperedge_to_node_layer_{}'.format(i))
            emb1_new = hyperedge_to_node(hyper_src_matmul, emb_edge_src)
            emb2_new = hyperedge_to_node(hyper_tgt_matmul, emb_edge_tgt)


            emb_src = emb1_new + emb_src
            emb_tgt = emb2_new + emb_src

        gt_1 = gt[0, 0:5, 0:5]
        gt_2 = gt[0, 5:(hyper_src.shape[2] - hyper_src.shape[1]), 5:(hyper_tgt.shape[2] - hyper_tgt.shape[1])]
        gt_3 = gt[0, (hyper_src.shape[2] - hyper_src.shape[1]):emb_edge_src.shape[1],
               (hyper_tgt.shape[2] - hyper_tgt.shape[1]):emb_edge_tgt.shape[1]]

        # source
        loss_s1_s2_s = 0
        for i in range(0, 5):
            t1 = torch.where(hyper_src[0, :, i] == 1)[0]
            t2 = torch.where(hyper_src[0, t1, 5:(hyper_src.shape[2] - hyper_src.shape[1])] == 1)
            t3 = torch.unique(t2[1])
            loss_s1_s2_s = loss_s1_s2_s + torch.sum(
                torch.square(emb_edge_src[:, i, :] - torch.sum(emb_edge_src[0, t3 + 5, :], dim=0)))

        loss_s2_s3_s = 0
        for i in range(5, (hyper_src.shape[2] - hyper_src.shape[1])):
            t1 = torch.where(hyper_src[0, :, i] == 1)[0]
            loss_s2_s3_s = loss_s2_s3_s + torch.sum(torch.square(
                emb_edge_src[:, i, :] - torch.sum(emb_edge_src[0, t1 + (hyper_src.shape[2] - hyper_src.shape[1]), :],
                                                  dim=0)))

        # tgt
        loss_s1_s2_t = 0
        for i in range(0, 5):
            t1 = torch.where(hyper_tgt[0, :, i] == 1)[0]
            t2 = torch.where(hyper_tgt[0, t1, 5:(hyper_tgt.shape[2] - hyper_tgt.shape[1])] == 1)
            t3 = torch.unique(t2[1])
            loss_s1_s2_t = loss_s1_s2_t + torch.sum(
                torch.square(emb_edge_tgt[:, i, :] - torch.sum(emb_edge_tgt[0, t3 + 5, :], dim=0)))


        loss_s2_s3_t = 0
        for i in range(5, (hyper_src.shape[2] - hyper_src.shape[1])):
            t1 = torch.where(hyper_tgt[0, :, i] == 1)[0]
            loss_s2_s3_t = loss_s2_s3_t + torch.sum(torch.square(
                emb_edge_tgt[:, i, :] - torch.sum(emb_edge_tgt[0, t1 + (hyper_tgt.shape[2] - hyper_tgt.shape[1]), :],
                                                  dim=0)))


        loss_1 = self.permutationLoss(s_s1[0], gt_1)
        loss_2 = self.permutationLoss(s_s2[0], gt_2)
        loss_3 = self.permutationLoss(s_s3[0], gt_3)

        loss = (0.1 * loss_1 + 0.2 * loss_2 + 0.7 * loss_3) + 0.001 * (loss_s1_s2_s + loss_s1_s2_t + loss_s2_s3_s + loss_s2_s3_t)


        scores2 = hungarian(s)
        tmp = gt + scores2
        acc = torch.where(tmp == 2)[0].shape[0] / torch.where(gt == 1)[0].shape[0]

        tmp2 = gt[0, (hyper_src.shape[2] - hyper_src.shape[1]):emb_edge_src.shape[1],
               (hyper_tgt.shape[2] - hyper_tgt.shape[1]):emb_edge_tgt.shape[1]] + scores2[0, (hyper_src.shape[2] -hyper_src.shape[1]): emb_edge_src.shape[1], (hyper_tgt.shape[2] -hyper_tgt.shape[1]):emb_edge_tgt.shape[1]]
        acc2 = torch.where(tmp2 == 2)[0].shape[0] / torch.where(
            gt[0, (hyper_src.shape[2] - hyper_src.shape[1]):emb_edge_src.shape[1],
            (hyper_tgt.shape[2] - hyper_tgt.shape[1]):emb_edge_tgt.shape[1]] == 1)[0].shape[0]

        return {
            'matches0': scores2[0],
            'loss': loss,
            'accuracy': acc,
            'accuracy2': acc2
        }

device = torch.device("cuda:0")





if __name__ == '__main__':

    with torch.cuda.device(0):
        setup_seed(114514)
        opt = parser.parse_args()


        files = []
        files += [opt.GT_train_path + f for f in os.listdir(opt.GT_train_path)]

        train_loader = torch.utils.data.DataLoader(dataset=files, shuffle=True, batch_size=opt.batch_size,
                                                   drop_last=True)

        hyperghmatching = HyperGHMatching().cuda().to(device)


        optimizer = torch.optim.Adam(hyperghmatching.parameters(), lr=opt.learning_rate, weight_decay=5e-4)
        mean_loss = []

        for epoch in range(1, opt.epoch + 1):
            epoch_loss = 0
            epoch_acc = 0
            hyperghmatching.train()
            print(epoch)

            for i, gt0_name in enumerate(train_loader):

                fixed = gt0_name[0][-31:-25]
                warp = gt0_name[0][-22:-16]

                # if fixed<=warp:
                #     continue

                fixed = [opt.feature_path + fixed + '.L.feat_node_edge_all.mat']
                warp = [opt.feature_path + warp + '.L.feat_node_edge_all.mat']

                data = load_data_aparc(fixed[0], warp[0], opt.GT_train_path)

                data1 = hyperghmatching(data)

                for k, v in data.items():
                    data[k] = v[0]
                data = {**data, **data1}

                Loss = data1['loss']
                epoch_loss += Loss.item()
                acc = data1['accuracy2']

                mean_loss.append(Loss)
                hyperghmatching.zero_grad()
                Loss.backward()
                optimizer.step()


                if (i + 1) % 10 == 0:
                    print('i [{}/{}], Loss: {:.4f}'
                          .format(i + 1, len(train_loader), torch.mean(torch.stack(mean_loss)).item()))
                    mean_loss = []
                    print(data['accuracy2'])

                if (i + 1) % 1000 == 0:
                    model_out_path = "model_epoch_{}_i_{}_our_0302_AB_1_0001.pth".format(epoch, i + 1)
                    torch.save(hyperghmatching, [opt.model_path + model_out_path][0])
                    print('Epoch [{}/{}], i [{}/{}], Checkpoint saved to {}'.format(epoch, opt.epoch, i + 1,
                                                                                    len(train_loader), model_out_path))
