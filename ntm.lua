require 'nn'
require 'nngraph'
require 'modules/Logging'
require 'modules/MulScalar'
require 'modules/Shifter'
require 'modules/PowScalar'
require 'modules/PowTable'
require 'modules/CircularConvolution'
require 'modules/SmoothCosineSimilarity'
require 'modules/ScalarMulTable'
require 'modules/OuterProd'

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
	self.initilizer = nil
	self:init_initializer()

	self.inputs = {}
	self.outputs = {}
	self.grad_inputs = {}

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

	local ctrl = nn.CAddTable()({in_h,r_h})

	local mem, r, wr = self:create_read_head(ctrl,prev_wr,prev_mem)

	local mem, ww = self:create_write_head(ctrl,prev_ww,prev_mem)

	local output = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.output_size)(ctrl))

	local inputs = {input, prev_mem, prev_ww, prev_r, prev_wr}
	-- local inputs = {input, prev_mem, prev_r}
	local outputs = {output, mem, ww, r, wr}


	nngraph.annotateNodes()
	self.ctrl = nn.gModule(inputs, outputs)
	self.ctrl.name = 'CTRLOL'

	graph.dot(self.ctrl.fg, 'Forward Graph', 'Forward Graph')

end

function NTM:create_read_head(h_state, prev_w, mem)

	local w = self:create_head(h_state, prev_w, mem)

	local r = nn.MixtureTable(1)({w,mem})

	return mem, r, w
end

function NTM:create_write_head(h_state, prev_w, mem)

	local w = self:create_head(h_state, prev_w, mem)

	local e = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local erasure = nn.AddConstant(1)(nn.MulConstant(-1)(nn.MM()({nn.View(self.mem_locations,1)(w),e}))) 

	local a = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local addition = nn.MM()({nn.View(self.mem_locations,1)(w),a})

	local new_mem = nn.CAddTable()({nn.CMulTable()({mem, erasure}),addition})
	nngraph.annotateNodes()
	return new_mem, w
end

-- function NTM:create_write_head(h_state, prev_w, mem)


-- 	-- local in_hidden = nn.View(self.hidden_state_size)(h_state)

-- 	local w = self:create_head(h_state, prev_w, mem)

-- 	local e = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
--  	local M_erase = nn.AddConstant(1)(nn.MulConstant(-1)(nn.MM()({nn.View(self.mem_locations,1)(w), e})))

-- 	local a = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
--     local M_write = nn.MM()({nn.View(self.mem_locations,1)(w), a})

--     local Mtilde = nn.CMulTable(){mem, M_erase}

-- 	-- write to memory
-- 	local M = nn.CAddTable(){Mtilde, M_write}

-- 	-- local erasure = nn.AddConstant(1)(nn.MulConstant(-1)(nn.MM()({nn.View(self.mem_locations,1)(w),e}))) 

-- 	-- local addition = nn.MM()({nn.View(self.mem_locations,1)(w),a})

-- 	-- local new_mem = nn.CAddTable()({nn.CMulTable()({mem, erasure}),addition})
-- 	nngraph.annotateNodes()
-- 	return M, w
-- end

function NTM:create_head(h_state, prev_w, mem)
	local in_hidden = nn.View(self.hidden_state_size)(h_state)

	-- key vector
  local k     = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(in_hidden))
  -- circular convolution kernel
  local s     = nn.SoftMax()(nn.Linear(self.hidden_state_size, #self.allowed_shifts)(in_hidden))
  -- weight sharpening parameter
  local beta  = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(in_hidden))
  -- gating parameter
  local g     = nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(in_hidden))
  -- exponential focusing parameter
  local gamma = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(in_hidden)))
  
  local sim = nn.SmoothCosineSimilarity(){mem, k}
  local wc = nn.SoftMax()(nn.ScalarMulTable(){sim, beta})
  local wg = nn.CAddTable(){
    nn.ScalarMulTable(){wc, g},
    nn.ScalarMulTable(){prev_w, nn.AddConstant(1)(nn.MulConstant(-1)(g))}
  }

  local wtilde = nn.CircularConvolution(){wg, s}
  local wpow = nn.PowTable(){wtilde, gamma}
  local w = nn.Normalize(1)(wpow)
	return w


end


-- function NTM:create_head(h_state, prev_w, mem)
-- 	local k_t = (nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state)))

-- 	local beta_t = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state))

-- 	local g_t = nn.Logging('gate',false)(nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(h_state)))

-- 	local s_t = nn.Logging('s_t', false)(nn.View(#self.allowed_shifts)(nn.SoftMax()(nn.Linear(self.hidden_state_size,#self.allowed_shifts)(h_state))))

-- 	local gamma_t = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state)))


-- 	local in_mem = nn.Identity()(mem)
-- 	local in_key = nn.Identity()(nn.View(self.mem_location_size)(k_t))

-- 	-- local dist = nn.CosineDistance()({in_mem,nn.Replicate(self.mem_locations)(in_key)})
-- 	local dist = nn.SmoothCosineSimilarity()({in_mem,in_key})

-- 	local w_c = nn.Logging('w_c',false)(nn.SoftMax()(nn.MulScalar()({beta_t,dist})))

-- 	local w_g1 = nn.MulScalar()({g_t, w_c})
-- 	local w_g2 = nn.MulScalar()({nn.AddConstant(1)(nn.MulConstant(-1)(g_t)), nn.Logging('prev',false)(prev_w)})

-- 	local w_g = nn.Logging('w_g',false)(nn.CAddTable()({w_g1,w_g2}))


-- 	local w_s = nn.Logging('w_s',false)(nn.CircularConvolution(self.allowed_shifts)({w_g, s_t}))

-- 	-- local w = nn.Logging('w',false)(nn.Normalize(1)(nn.PowScalar()({gamma_t, w_s})))
-- 	local w = nn.Logging('w',false)(nn.Normalize(1)(nn.PowTable()({w_s, nn.View(1)(gamma_t)})))
-- 	-- local w = nn.Logging('w',false)(nn.SoftMax()( w_s))

	

-- 	-- local new_w = nn.Linear(self.hidden_state_size, self.mem_locations)(h_state)

-- 	-- local r = nn.MixtureTable()({new_w,new_mem})

-- 	nngraph.annotateNodes()
-- 	return w


-- end

function NTM:init_initializer()
	local input = nn.Identity()()

	local mem = nn.View(self.mem_locations,self.mem_location_size)(nn.Linear(1,self.mem_locations * self.mem_location_size)(input))
	local wr = nn.SoftMax()(nn.Linear(1,self.mem_locations)(input))
	local ww = nn.SoftMax()(nn.Linear(1,self.mem_locations)(input))
	local r = nn.Tanh()(nn.Linear(1,self.mem_location_size)(input))



	self.initilizer = nn.gModule({input},{nn.Identity()(input),mem,ww,r,wr})
end

function NTM:getFirstGradOutputs()

	local mem_grad = torch.zeros(self.mem_locations, self.mem_location_size)
	local ww_grad = torch.zeros(self.mem_locations)
	local r_grad = torch.zeros(self.mem_location_size)
	local wr_grad = torch.zeros(self.mem_locations)
	return {0, mem_grad, ww_grad, r_grad, wr_grad}

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

	local inputs
	if self.sequence_step == 1 then
		inputs = self.initilizer:forward(torch.Tensor{0})
	else
		inputs = copy_table(self.outputs[self.sequence_step - 1])
	end
	inputs[1] = input

	self.inputs[self.sequence_step] = copy_tensor_table(inputs)
	self.outputs[self.sequence_step] = copy_tensor_table(self.ctrl:forward(inputs))

	return self.outputs[self.sequence_step][1]
end

function print_table(table)
	for i=1,#table do
		if(type(table[i]) == 'table') then
			print('ELEM ' .. i)
			print_table(table[i])
		else
			print(table[i])
		end
	end
end

function max_table(table)
	local max = -1
	for i=2,#table do
		local cur_max = table[i]:max()
		if max < cur_max then
			max = cur_max
		end
	end
	return max
end

function test_nan(table)
	for i=1,#table do
		if table[i]:max()~=table[i]:max() then
			print(i)
			print(table[i])
			print_table(table)
			return true
		end
	end
	return false
end


function NTM:backward(input, gradOutput)
	

	-- if self.sequence_step == 0 then
	-- 	inputs = self.initilizer:forward(torch.Tensor{0})
	-- else
	-- 	inputs = copy_table(self.outputs[self.sequence_step])
	-- end
	-- inputs[1] = input

	local inputs = self.inputs[self.sequence_step]

	local grad_outputs = self.grad_inputs[self.sequence_step+1]
	if grad_outputs == nil then
		self.grad_inputs[self.sequence_step+1] = self:getFirstGradOutputs()
		grad_outputs = self.grad_inputs[self.sequence_step+1]
	-- else
	-- 	grad_outputs = copy_tensor_table(self.grad_inputs)
	end

	grad_outputs[1] = gradOutput

	local grad_inputs = self.ctrl:backward(inputs, grad_outputs)

	local stop = test_nan(grad_inputs)
	
	if stop then
		print('Got nan in step ' .. self.sequence_step )
		print('In :')
		print(input)
		print('Inputs :')
		print_table(inputs)
		print('grad_outputs')
		print_table(grad_outputs)
		-- error('Need to stop ')
	end



	if self.sequence_step == 1 then
		grad_inputs[1] = torch.Tensor{0}
		self.initilizer:backward(torch.Tensor{0},grad_inputs)
	end

	self.grad_inputs[self.sequence_step] = grad_inputs

	self.sequence_step = self.sequence_step - 1

	return self.grad_inputs[self.sequence_step + 1][1], stop
end

function NTM:parameters()
	ctrl_p, ctrl_g = self.ctrl:parameters()
	init_p, init_g = self.initilizer:parameters()

	print('Ntm net has ')
	print(self.ctrl:getParameters():size())
	print('Init net has ')
	print(self.initilizer:getParameters():size())

	tablex.insertvalues(ctrl_p, init_p)
	tablex.insertvalues(ctrl_g, init_g)

	return ctrl_p, ctrl_g
end

function NTM:zeroGradParameters()
	self.ctrl:zeroGradParameters()
	-- local params, grads = self.initilizer:getParameters() 
	self.initilizer:zeroGradParameters()
end

