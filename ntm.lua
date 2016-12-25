require 'nn'

function createSample(sampleSize)
	local result = torch.Tensor():rand(sampleSize):gt(0.5):double()

	return torch.Tensor():rand(sampleSize):gt(0.5):double()
end

function createCopyDataSet(nSample, sampleSize)
	local dataSet = torch.Tensor(nSample, sampleSize)
	for i=1,nSample do
		dataSet[{{i},{}}] = createSample(sampleSize) 
	end
	return dataSet, dataSet
end

xs,ys = createCopyDataSet(10,4)

print(xs)	
print(ys)
print(xs)	
print(ys)	