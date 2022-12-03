import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.transforms as transform

from densefusion_lib.network import ModifiedResnet, PoseNetFeat

from lib.feature_extractor import FeatureExtractor




class ScalePoseNet(nn.Module):

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

class LSTMPose(nn.Module):
	def __init__(self, n_feat, out_dim, resize=(32, 48)):

		super(LSTMPose, self).__init__()

		if ((resize[0] % 4 != 0) and (resize[1] % 4 !=0)):
			raise Exception("H and W must be divisible by 4")
		self.feat = FeatureExtractor()
		self.LSTM = nn.LSTM(input_size=n_feat, hidden_size=512, num_layers=3)
		self.out = nn.LazyLinear(out_dim)

		# pc and rgb crops will be resized to this in order to have common HxW
		self.extracted_feature_size = (resize[0], resize[1])
		self.resizer = transform.Resize(self.extracted_feature_size)

	def forward(self, img, pc):
		"""
		@param img: color img of size Nx3xHxW
		@param pc: pc of size Nx3xHxW
		"""
		color_feat, geo_feat = self.feat(img,pc) #each of size Nx3xHxW
		all_feats = torch.cat([color_feat,geo_feat], 2) #Nx64xHxW

		all_feats = self.resizer(all_feats) #Nx64x H_common x W_common

		_,_,H,W = all_feats.shape

		chunk_H = H//4
		chunk_W = W//4

		return pred_verts

class PCNet(nn.Module):
	def __init__(self, inchannels, emb_dim=64, out_dim=3072):
		super(PCNet, self).__init__()
		self.net = nn.Sequential(
			torch.nn.Conv2d(inchannels, 512, 5, padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(512, 256, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(256, 128, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			nn.Linear(out_dim, emb_dim)
			# nn.Linear(512, emb_dim)
		)

	def forward(self, x):
		return self.net(x)


class ColorNet(nn.Module):
	def __init__(self,  inchannels, emb_dim=64):
		super(ColorNet, self).__init__()
		self.net = nn.Sequential(
			torch.nn.Conv2d(inchannels, 64, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(64, 128, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(128, 64, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(64, 32, 3, padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			nn.Linear(192, emb_dim)
		)
	def forward(self, x):
		return self.net(x)

class GlobNet(nn.Module):
	def __init__(self, infeatures, outfeatures = 24):
		super(GlobNet, self).__init__()
		self.net = nn.Linear(infeatures, outfeatures)

	def forward(self, x):
		return self.net(x)

		return pred_verts


