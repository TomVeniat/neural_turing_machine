require 'nn'
require 'Memory'

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




m = Memory(5,2)
print(m.mem)

-- r = torch.Tensor({0.9,0.1,0,0,0})
-- print(r:resize(1,5))
-- print(m:read(r))


r = torch.Tensor({-2,1}):resize(1,2)
print(m:getContentWeightings(r))


xs,ys = createCopyDataSet(10,4)
