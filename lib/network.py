import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from densefusion_lib.network import ModifiedResnet, PoseNetFeat



class ScalePoseNet(nn.module):
	def __init__(self, num_points, num_obj):
		super(ScalePoseNet, self).__init__()
		self.num_points = num_points
		self.cnn = ModifiedResnet()
		self.feat = PoseNetFeat(num_points)

		self.net = nn.Sequential(
			nn.Conv1d(1408,640,1),
			nn.ReLU(),
			nn.Conv1d(640,256,1),
			nn.ReLU(),
			nn.Conv1d(256,128,1),
			nn.ReLU(),
			nn.Flatten(),
			nn.LazyLinear(2048),
			nn.ReLU(),
			nn.Linear(2048,512),
			nn.ReLU(),
			nn.Linear(512,8*3) #3 xyz  values for each 3D bounding box, 8 vertices
			)


	def forward(self, img, pc, choose):
		# img: input image
		# pc: point_cloud points
		# choose: which points to use
		
		out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        
        pc = pc.transpose(2, 1).contiguous()
        ap_pc = self.feat(pc, emb)

        pred_verts = self.net(ap_pc)

        return pred_verts