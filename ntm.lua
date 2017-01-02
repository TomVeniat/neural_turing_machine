require 'nn'
require 'Memory'

function createSample(sampleSize)
	local result = torch.Tensor():rand(sampleSize):gt(0.5):double()

	return torch.Tensor():rand(sampleSize):gt(0.5):double()
end

function createCopyDataSet(nSample, sampleSize)
	local x = torch.zeros(nSample+2, sampleSize+2)
	local y = torch.zeros(nSample, sampleSize)
	for i=1,nSample do
		local sample = createSample(sampleSize)
		x[{{i+1},{}}] = torch.cat(sample, torch.Tensor({0,0}))
		y[{{i},{}}] = sample
	end

	x[{{1},{sampleSize + 1}}] = 1
	x[{{nSample+2},{sampleSize + 2}}] = 1

	return x, y
end




m = Memory(5,2)
print(m.mem)

r = torch.Tensor({0.9,0.1,0,0,0})
print(r:resize(1,5))
print(m:read(r))


r = torch.Tensor({1,2}):resize(1,2)
print(m:getContentWeightings(r,120))


xs,ys = createCopyDataSet(10,4)

print(xs)
print(ys)
