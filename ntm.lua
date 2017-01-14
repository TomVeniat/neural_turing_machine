require 'nn'
require 'nngraph'
require 'Memory'
require 'modules/Logging'
require 'modules/MulScalar'
require 'modules/ConcatTensor'
require 'modules/Shifter'
require 'modules/PowScalar'
require 'modules/Resizer'

require 'modules/Hijack'

local NTM, Parent = torch.class('nn.NTM', 'nn.Module')

function NTM:__init( params)
	Parent.__init(self)

	print ('INIT ntm')
	nngraph.setDebug(true)
	self.input_size = params.input_size or 3
	self.output_size = params.output_size or 3
	self.mem_locations = params.mem_locations or 50
	self.mem_location_size = params.mem_location_size or 3
	self.hidden_state_size = params.hidden_state_size or 80
	self.allowed_shifts = params.allowed_shifts or {-1,0,1}

	self.mem = torch.range(1, self.mem_locations * self.mem_location_size):resize(self.mem_locations, self.mem_location_size)

	self.controller = nil
	self:init_controller()

	self.outputs = {}

	self.sequence_step = 0

end

function NTM:init_controller()
	local input = nn.Identity()()

	local prev_mem = nn.Identity()()
	local prev_wr = nn.Identity()()
	local prev_ww = nn.Identity()()

	local prev_r = nn.Identity()()

	local in_h = nn.Linear(self.input_size, self.hidden_state_size)(input)
	local r_h = nn.Linear(self.mem_location_size, self.hidden_state_size)(prev_r)

	local ctrl = nn.Linear(self.hidden_state_size, self.hidden_state_size)(nn.CAddTable()({in_h,r_h}))

	local mem, r, wr = self:create_read_head(ctrl,prev_wr,prev_mem)

	local mem, ww = self:create_write_head(ctrl,prev_ww,prev_mem)

	local output = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.output_size)(ctrl))

	local inputs = {input, prev_mem, prev_ww, prev_r, prev_wr}
	-- local inputs = {input, prev_mem, prev_r}
	local outputs = {output, mem, ww, r, wr}


	nngraph.annotateNodes()
	self.ctrl = nn.gModule(inputs, outputs)
	self.ctrl.name = 'gra'

	graph.dot(self.ctrl.fg, 'Forward Graph', 'Forward Graph')

end

function NTM:create_read_head(h_state, prev_w, mem)

	local w = self:create_head(h_state, prev_w, mem)

	local r = nn.Logging('read',false)(nn.MixtureTable(1)({w,mem}))

	return mem, r, w
end

function NTM:create_write_head(h_state, prev_w, mem)

	local w = self:create_head(h_state, prev_w, mem)

	local e = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local erasure = nn.AddConstant(1)(nn.MulConstant(-1)(nn.MM()({nn.Resizer({self.mem_locations,1})(w),e}))) 

	local a = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local addition = nn.MM()({nn.Resizer({self.mem_locations,1})(w),a})

	local new_mem = nn.CAddTable()({nn.CMulTable()({mem, erasure}),addition})
	nngraph.annotateNodes()
	return new_mem, w
end

function NTM:create_head(h_state, prev_w, mem)
	local k_t = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))

	local beta_t = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state))

	local g_t = nn.Logging('gate',false)(nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(h_state)))

	local s_t = nn.Hijack(torch.Tensor{0,0,1})(nn.SoftMax()(nn.Linear(self.hidden_state_size,#self.allowed_shifts)(h_state)))

	local gamma_t = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state)))


	local in_mem = nn.Identity()(mem)
	local in_key = nn.Identity()(k_t)

	local dist = nn.CosineDistance()({in_mem,nn.ConcatTensor(self.mem_locations)(in_key)})

	local w_c = nn.Logging('w_c',false)(nn.SoftMax()(nn.MulScalar()({beta_t,dist})))

	-- local gt_c = nn.
	local w_g1 = nn.MulScalar()({g_t, w_c})
	local w_g2 = nn.MulScalar()({nn.AddConstant(1)(nn.MulConstant(-1)(g_t)), nn.Logging('prev',false)(prev_w)})

	local w_g = nn.Logging('w_g',false)(nn.CAddTable()({w_g1,w_g2}))

	local w_s = nn.Logging('w_s',false)(nn.Shifter(self.allowed_shifts)({w_g, s_t}))

	local w = nn.Logging('w',false)(nn.PowScalar()({gamma_t, w_s}))

	

	-- local new_w = nn.Linear(self.hidden_state_size, self.mem_locations)(h_state)

	-- local r = nn.MixtureTable()({new_w,new_mem})

	nngraph.annotateNodes()
	return w
	-- return new_mem, new_w, r


end

function NTM:getFirstInputs()

	local wr = nn.SoftMax():forward(torch.rand(self.mem_locations))
	local ww = nn.SoftMax():forward(torch.rand(self.mem_locations))
	local lin = nn.Linear(1,self.mem_location_size):forward(torch.Tensor({0}))
	local r = nn.Tanh():forward(lin)
	return {0, self.mem, ww,r, wr}
	-- return {0, self.mem, r}

end

function copy_table(tab)
	local res = {}
	for i,v in ipairs(tab) do
		res[i] = v
	end
	return res
end

function copy_tensor_table(tab)
	local res = {}
	for i=1,#tab do
		res[i] = tab[i]:clone()
	end
	return res
end

function NTM:forward(input)

	self.sequence_step = self.sequence_step + 1

	if self.sequence_step == 1 then
		inputs = self:getFirstInputs()
	else
		inputs = copy_table(self.outputs[self.sequence_step - 1])
	end
	inputs[1] = input

	self.outputs[self.sequence_step] = copy_tensor_table(self.ctrl:forward(inputs))

	return self.outputs[self.sequence_step][1]
end

function NTM:backward(input, gradOutput)

	self.sequence_step = self.sequence_step - 1

	
	return nil
end
