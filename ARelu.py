import torch

def ARELU(act):
	act = act.type(torch.FloatTensor)
	k = 0.5
	n = 1.1
	x = torch.tensor(0.0)
	y = k*torch.pow(act,n)
	t = torch.where(act<0,x,y)
	return t
