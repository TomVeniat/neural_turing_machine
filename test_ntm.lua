require 'nn'
require 'nngraph'
require 'Memory'
require 'ntm'

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
print(m.mem)

print()


s = {}
s[0] = 0.05
s[-1] = 0.9
s[-2] = 0.05

r = torch.Tensor({0,0.1,0.8,0.1,0}):resize(1,5)

shifted = m:shiftWeigthing(r, s)
print(shifted)


r = torch.Tensor({0,0.1,0.8,0.1,0}):resize(1,5)

sharpened = m:sharpenWeigths(r,2)

print(sharpened)

e = torch.Tensor({0,1}):resize(1,2)

print(m.mem)
m:erase(r:t(),e)
print(m.mem)
m:add(r:t(),e)
print(m.mem)

a = torch.Tensor({0,1,1,0,1})
b = torch.Tensor({0,0,1,0,1})
c = torch.Tensor({0,1,1,0,0})
d = torch.Tensor({-700,12,123,-14,6})
e = torch.Tensor({-700,-45,123,-14,6})


criterion = nn.BCECriterion()


sep = '-'
print(sep:rep(30))

params = {
	input_size = 4,
	output_size = 5,
	mem_locations = 50,
	mem_location_size = 2,
	hidden_state_size = 80,
	allowed_shifts = {-1,0,1}
}

nt = nn.NTM(nil,nil,params)

print(nt.mem)

-- nngraph.annotateNodes()

t = 100

for i=1,t do
	local sample = createSample(3)
end

print(nt:forward(torch.Tensor{1,2,3,4}:resize(1,4)))