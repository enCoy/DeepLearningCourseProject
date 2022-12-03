import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from densefusion_lib.network import ModifiedResnet, PoseNetFeat


'''
This class extracts color and geometry feature embeddings from RGB image and Point Cloud. 
It also creates a global feature.
'''

class FeatureExtractor(nn.Module):
    def __init__(self, num_points=1024, num_obj=7):
        super(FeatureExtractor, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet() #model that generates color features
        self.feat = PoseNetFeat(num_points)

        self.pc_feat = nn.Sequential(
            nn.Conv2d(3,16,1),
            nn.Conv2d(16,32,1)
        )

        #model that generates features from depth image
        self.depth_feat = nn.Sequential(
            nn.Conv2d(1,3,kernel_size=1),
            nn.Conv2d(3,64,1),
            nn.Conv2d(64,128,1),
            nn.Conv2d(128,1024,1)
        )



    def forward(self, img, pc, choose=None):
        """
        Generate all features.
        @param img: img crop to get color features for, shape: Nx3xHxW
        @param pc: masked point cloud points for object, shape: Nx3xHxW
        @param choose: what points to choose for embedding
        @return: tuple of feature embeddings (color_img_feat, pc_feats)
        """
        N,_,H,W = img.shape

        color_emb = self.cnn(img)  # shape (N,32,H,W)
        geo_emb = self.pc_feat(pc) # shape (N,32,H,W)

        
        return (color_emb, geo_emb)

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

    def depth_emb(self, dep_img):
        """

        @param dep_img: depth image of size HxWx1
        @return: depth image features
        """
        return self.depth_feat(dep_img)