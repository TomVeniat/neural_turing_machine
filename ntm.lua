require 'nn'
require 'nngraph'
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



local NTM, Parent = torch.class('nn.NTM', 'nn.Module')

function NTM:__init(nInput, nOutput)
	Parent.__init(self)

	print ('INIT ntm')
	nngraph.setDebug(true)
	self.input_size = 3
	self.output_size = 3
	self.mem_locations = 10
	self.mem_location_size = 5
	self.hidden_state_size = 100
	self.allowed_shifts = {-1,0,1}

	self.mem = torch.range(1, self.mem_locations * self.mem_location_size):resize(self.mem_locations, self.mem_location_size)

	self.controller = nil
	self:init_controller()

	self.outputs = {}

	self.seq_step = 0


end

function NTM:init_controller()
	local input = nn.Identity()()

	local prev_mem = nn.Identity()()
	local prev_wr = nn.Identity()()
	local prec_ww = nn.Identity()()

	local prev_r = nn.Identity()()

	local in_h = nn.Linear(self.input_size, self.hidden_state_size)(input)
	local r_h = nn.Linear(self.mem_location_size, self.hidden_state_size)(prev_r)

	local ctrl = nn.Linear(self.hidden_state_size, self.hidden_state_size)(nn.CAddTable()({in_h,r_h}))

	local mem, wr, r = self:create_head(ctrl,prev_wr,prev_mem)

	local output = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.output_size)(ctrl))

	-- local inputs = {input, prev_mem, prev_wr, prev_r}
	local inputs = {input, prev_mem, prev_r}
	local outputs = {output, mem, wr, r}


	nngraph.annotateNodes()
	self.ctrl = nn.gModule(inputs, outputs)

end

function NTM:create_head(h_state, prev_w, mem)
	-- k_t = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))

	-- beta_t = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state))

	-- g_t = nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(h_state))

	-- s_t = nn.SoftMax()(nn.Linear(self.hidden_state_size,#self.allowed_shifts)(h_state))

	-- gamma_t = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state)))


	-- local in_mem = nn.Identity()(mem)
	-- local in_key = nn.Identity()(k_t)

	-- local dimensions = {}

	-- for i=1,self.mem_locations do
	-- 	dimensions[i] = nn.Identity()(in_key)
	-- end

	-- local full_k = nn.JoinTable(1)(dimensions)


	-- local dist = nn.CosineDistance()({in_mem,full_k})
	-- local w_c = nn.SoftMax()(nn.CMulTable()({beta_t,dist}))

	-- local r = nn.MixtureTable()({w_c,in_mem})

	local new_mem = nn.Identity()(mem)

	local new_w = nn.Linear(self.hidden_state_size, self.mem_locations)(h_state)

	local r = nn.MixtureTable()({new_w,new_mem})

	-- return new_mem, w_c, r
	return new_mem, new_w, r

	-- return mem, prev_w, nn.MixtureTable()({w_c,in_mem})

end

function NTM:getFirstInputs()

	local wr = nn.SoftMax():forward(torch.rand(self.mem_locations))
	local lin = nn.Linear(1,self.mem_location_size):forward(torch.Tensor({0}))
	local r = nn.Tanh():forward(lin)
	-- return {0, self.mem, wr, r}
	return {0, self.mem, r}

end

function NTM:forward(input)

	self.seq_step = self.seq_step + 1

	local inputs 
	if self.seq_step == 1 then
		inputs = self:getFirstInputs()
	else
		inputs = self.outputs[self.seq_step - 1]
	end
	inputs[1] = input

	self.outputs[self.seq_step] = self.ctrl:forward(inputs)

	return self.outputs[self.seq_step][1]
end

nt = nn.NTM()

print(nt.mem)
print(nt.ctrl)

-- nngraph.annotateNodes()
-- graph.dot(nt.ctrl, 'Forward Graph')
print(nt:forward(torch.Tensor{1,2,3}))
