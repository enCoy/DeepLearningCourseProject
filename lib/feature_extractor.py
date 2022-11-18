import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from densefusion_lib.network import ModifiedResnet, PoseNetFeat


'''
This class extracts color and geometry feature embeddings from RGB image and Point Cloud. 
It also creates a global feature.
'''
class FeatureExtractor(nn.module):
    def __init__(self, num_points, num_obj):
        super(FeatureExtractor, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet() #model that generates color features
        self.feat = PoseNetFeat(num_points)

        self.pc_feat = #TODO set the model that generates the geometric features here



    def forward(self, img, pc, choose):
        """
        Generate all features.
        @param img: img crop to get color features for
        @param pc: masked point cloud points for object
        @param choose: what points to choose for embedding
        @return: concatenated feature vector [color_features,geometric_features,global_feature] of size len(choose)
        """
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        color_emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        color_emb = torch.gather(color_emb, 2, choose).contiguous()

        pc = pc.transpose(2, 1).contiguous()

        # TODO: once we change the model for geometric feature embedding, change the way global features are generated
        all_feats = self.feat(pc, color_emb)  #[color_feat, geo_feat, global_feat]

        return all_feats

    def color_emb(self, img):
        """
        @param img: img crop to generate color features for size HxWx3
        @return: color feature embedding, size HxWxD
        """
        return self.cnn(img)

    def geo_emb(self, pc):
        """
        Returns the geometric feature embedding for pc
        @param pc: pointcloud points
        @return: geometric feature embedding calculated by pc_feat model.
        """
        return self.pc_feat(pc)