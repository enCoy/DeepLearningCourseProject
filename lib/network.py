import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.transforms as transform

from densefusion_lib.network import ModifiedResnet, PoseNetFeat

from lib.feature_extractor import FeatureExtractor

from lib.convlstm import ConvLSTM



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
	def __init__(self, out_dim=24, resize=(36, 36), device = "cpu"):

		super(LSTMPose, self).__init__()

		if ((resize[0] % 4 != 0) and (resize[1] % 4 !=0)):
			raise Exception("H and W must be divisible by 4")

		self.feat = FeatureExtractor()
		self.feat.to(device)
		self.encoder = nn.Sequential(
			nn.Conv2d(64,32,1),
			nn.Conv2d(32,8,1)
		)
		self.encoder.to(device)
		self.LSTM = ConvLSTM(input_dim=8, hidden_dim=2, kernel_size=(1, 1), num_layers=2, batch_first=True)
		self.flatten = nn.Flatten()
		self.out = nn.LazyLinear(out_dim)

		# pc and rgb crops will be resized to this in order to have common HxW
		self.extracted_feature_size = (resize[0], resize[1])
		self.resizer = transform.Resize(self.extracted_feature_size)

		#number of horizontal and vertical patches to send to LSTM
		self.n_H = 4
		self.n_V = 4


	def forward(self, img, pc):
		"""
		@param img: color img of size Nx3xHxW
		@param pc: pc of size Nx3xHxW
		"""

		color_feat, geo_feat = self.feat(img,pc) #each of size Nx3xHxW

		color_feat = self.resizer(color_feat)
		geo_feat = self.resizer(geo_feat)

		all_feats = torch.cat([color_feat,geo_feat], 1) #Nx64xH_commonxW_common
		encoded = self.encoder(all_feats) #N x 8 x H_common x W_common

		N,C,H,W = encoded.shape

		chunk_H = H // self.n_H
		chunk_W = W // self.n_V

		num_patches = self.n_V*self.n_H

		patches = encoded.unfold(1,C,C).unfold(2,chunk_W, chunk_W).unfold(3,chunk_H,chunk_H)
		patches = patches.reshape((N, num_patches, C, chunk_W, chunk_H))

		last_layer_output, last_state_list = self.LSTM(patches)

		last_seq_out = last_layer_output[0][:,-1,:,:,:] #final output from lstm
		last_seq_out = self.flatten(last_seq_out)

		out = self.out(last_seq_out)

		return out

class PCNet(nn.Module):
	def __init__(self, inchannels, emb_dim=64):
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
			nn.Linear(512, emb_dim)
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


