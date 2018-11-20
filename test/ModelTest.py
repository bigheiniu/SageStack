from script.aggregators import MeanAggregator
import torch
def aggTest():
    mean = MeanAggregator(1, 300, 300)
    #batch first
    vecs = torch.rand(10,300,requires_grad=True)
    nei = torch.rand(10,4,300,requires_grad=True)
    result = mean.forward(vecs,nei,'relu')
    print(mean.parameters())
    print(result)
def unSuperviseTest():


if __name__ == '__main__':
    aggTest()