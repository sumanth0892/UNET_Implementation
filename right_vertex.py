import torch
import torch.nn as nn

class UpConvolution(nn.Module):
	def __init__(self,input_channels,output_channels):
		super(UpConvolution,self).__init__()
		self.conv1 = nn.Conv2d(input_channels,output_channels,(3,3))
		self.conv2 = nn.Conv2d(output_channels,output_channels,(3,3))
		self.convtranspose = nn.ConvTranspose2d(output_channels,output_channels//2,(2,2),(2,2))
		self.Relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.Relu(x)
		x = self.conv2(x)
		x = self.Relu(x)
		x = self.dropout(x)
		x = self.convtranspose(x)
		return x

if __name__ == '__main__':
	inp = torch.rand(4,512,104,104)
	out = up_conv(inp)
	print(out.size())
