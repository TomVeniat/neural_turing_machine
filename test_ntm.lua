require 'nn'
require 'nngraph'
require 'Memory'
require 'ntm'

function createSample(sampleSize, start_tag, end_tag)
	-- local result = torch.Tensor():rand(sampleSize):gt(0.5):double()
	local res
	if start_tag or end_tag then
		res = torch.zeros(unpack(sampleSize))
	else 
		res = torch.Tensor():rand(unpack(sampleSize)):gt(0.5):double()
	end

	res[1][-2] = start_tag and 1 or 0
	res[1][-1] = end_tag and 1 or 0

	return res
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




-- m = Memory(5,2)
-- print(m.mem)

-- r = torch.Tensor({0.9,0.1,0,0,0})
-- print(r:resize(1,5))
-- print(m:read(r))


-- r = torch.Tensor({1,2}):resize(1,2)
-- print(m:getContentWeightings(r,120))


-- xs,ys = createCopyDataSet(10,4)

-- print(xs)
-- print(ys)
-- print(m.mem)

-- print()


-- s = {}
-- s[0] = 0.05
-- s[-1] = 0.9
-- s[-2] = 0.05

-- r = torch.Tensor({0,0.1,0.8,0.1,0}):resize(1,5)

-- shifted = m:shiftWeigthing(r, s)
-- print(shifted)


-- r = torch.Tensor({0,0.1,0.8,0.1,0}):resize(1,5)

-- sharpened = m:sharpenWeigths(r,2)

-- print(sharpened)

-- e = torch.Tensor({0,1}):resize(1,2)

-- print(m.mem)
-- m:erase(r:t(),e)
-- print(m.mem)
-- m:add(r:t(),e)
-- print(m.mem)

-- a = torch.Tensor({0,1,1,0,1})
-- b = torch.Tensor({0,0,1,0,1})
-- c = torch.Tensor({0,1,1,0,0})
-- d = torch.Tensor({-700,12,123,-14,6})
-- e = torch.Tensor({-700,-45,123,-14,6})


-- criterion = nn.BCECriterion()


sep = '-'
print(sep:rep(30))

params = {
	input_size = 7,
	output_size = 7,
	mem_locations = 10,
	mem_location_size = 15,
	hidden_state_size = 80,
	allowed_shifts = {-1,0,1}
}

nt = nn.NTM(params)


-- nngraph.annotateNodes()

-- xs, ys = createCopyDataSet(50,3)

local t = 5
local inputs = {}
local outputs = {}
for i=1,t do
	local sample = createSample({1,7},false, false)
	inputs[i] = sample
	outputs[i] = nt:forward(sample)
	-- outputs[i] = nt:forward(torch.Tensor{1,2,3,4,5}:resize(1,5))
end

for i=1,#outputs do
	print(inputs[i])
end
for i=1,#outputs do
	print(nt.outputs[i][1])
end

print(nt.outputs)
