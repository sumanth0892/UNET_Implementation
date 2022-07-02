import torch
import torch.nn as nn

class SimpleConvolution(nn.Module):
	def __init__(self,input_channels,output_channels):
		super(SimpleConvolution,self).__init__()
		self.conv1 = nn.Conv2d(input_channels,output_channels,(3,3))
		self.conv2 = nn.Conv2d(output_channels,output_channels,(3,3))
		self.Relu = nn.ReLU()
		self.dropout = nn.Dropout2d(0.2)

	def forward(self,x):
		x = self.conv1(x)
		x = self.Relu(x)
		x = self.conv2(x)
		x = self.Relu(x)
		x = self.dropout(x)
		return x

if __name__ == '__main__':
	simple_conv = SimpleConvolution(1,64)
	inp = torch.rand(4,1,572,572)
	out = simple_conv(inp)
	print(out.size())