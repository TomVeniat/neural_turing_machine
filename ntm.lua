require 'nn'
require 'nngraph'
require 'modules/Logging'
require 'modules/MulScalar'
require 'modules/Shifter'
require 'modules/PowScalar'
require 'modules/SmoothCosineSimilarity'

require 'modules/Hijack'

local NTM, Parent = torch.class('nn.NTM', 'nn.Module')

function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

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

	self.ctrl={}
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

	local mem_red,r, wr = self:create_read_head(ctrl,prev_wr,prev_mem)

	local new_mem, ww = self:create_write_head(ctrl,prev_ww,mem_red)

	local output = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.output_size)(ctrl))

	local inputs = {input, prev_mem, prev_ww, prev_r, prev_wr}

	local outputs = {output, new_mem, ww, r, wr}


	nngraph.annotateNodes()
	self.ctrl[1] = nn.gModule(inputs, outputs)
	self.ctrl.name = 'CTRLOL'

	graph.dot(self.ctrl[1].fg, 'Forward Graph', 'Forward Graph')

end

function NTM:create_read_head(h_state, prev_w, memoire)

	local w = self:create_head(h_state, prev_w, memoire)

	local r = nn.MixtureTable()({w,memoire})
	nngraph.annotateNodes()
	return memoire, r, w
end

function NTM:create_write_head(h_state, prev_w, memoise)

	local w = self:create_head(h_state, prev_w, memoise)

	local e = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local erasure = nn.AddConstant(1)(nn.MulConstant(-1)(nn.MM()({nn.View(self.mem_locations,1)(w),nn.View(1, self.mem_location_size)(e)}))) 

	local a = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state))
	local addition = nn.MM()({nn.View(self.mem_locations,1)(w),nn.View(1, self.mem_location_size)(a)})

	local new_mem = nn.CAddTable()({nn.CMulTable()({memoise, erasure}),addition})
	
	return new_mem, w
end

function NTM:create_head(h_state, prev_w, mem)
	local k_t = (nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(h_state)))

	local beta_t = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state))

	local g_t = nn.Logging('gate',false)(nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(h_state)))

	local s_t = nn.Logging('s_t', false)(nn.View(#self.allowed_shifts)(nn.SoftMax()(nn.Linear(self.hidden_state_size,#self.allowed_shifts)(h_state))))

	local gamma_t = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(h_state)))


	local in_mem = nn.Identity()(mem)
	local in_key = nn.Identity()(nn.View(self.mem_location_size)(k_t))

	-- local dist = nn.CosineDistance()({in_mem,nn.Replicate(self.mem_locations)(in_key)})
	local dist = nn.SmoothCosineSimilarity()({in_mem,in_key})

	local w_c = nn.Logging('w_c',false)(nn.SoftMax()(nn.MulScalar()({beta_t,dist})))

	local w_g1 = nn.MulScalar()({g_t, w_c})
	local w_g2 = nn.MulScalar()({nn.AddConstant(1)(nn.MulConstant(-1)(g_t)), nn.Logging('prev',false)(prev_w)})

	local w_g = nn.Logging('w_g',false)(nn.CAddTable()({w_g1,w_g2}))


	local w_s = nn.Logging('w_s',false)(nn.Shifter(self.allowed_shifts)({w_g, s_t}))

	local w = nn.Logging('w',false)(nn.Normalize(1)(nn.PowScalar()({gamma_t, w_s})))


	nngraph.annotateNodes()
	return w
end

function NTM:init_initializer()
	local input = nn.Identity()()

	local mem = nn.View(self.mem_locations,self.mem_location_size)(nn.Linear(1,self.mem_locations * self.mem_location_size)(input))
	local wr_lin = nn.Linear(1,self.mem_locations)
	wr_lin.bias:copy(torch.range(self.mem_locations, 1, -1))
	local wr = nn.SoftMax()(wr_lin(input))

	local ww_lin = nn.Linear(1,self.mem_locations)
	ww_lin.bias:copy(torch.range(self.mem_locations, 1, -1))
	local ww = nn.SoftMax()(ww_lin(input))
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

function NTM:forward(input)

	self.sequence_step = self.sequence_step + 1

	local ctrol = self.ctrl[self.sequence_step]
	if ctrol == nil then
		ctrol = clone_many_times(self.ctrl[1],2)[2]
		self.ctrl[self.sequence_step] = ctrol
	end


	local inputs
	if self.sequence_step == 1 then
		inputs = self.initilizer:forward(torch.Tensor{0})
	else
		inputs = self.outputs[self.sequence_step - 1]
	end
	inputs[1] = input

	self.inputs[self.sequence_step] = inputs
	self.outputs[self.sequence_step] = ctrol:forward(inputs)

	return self.outputs[self.sequence_step][1]
end

function NTM:backward(input, gradOutput)
	
	local inputs = self.inputs[self.sequence_step]

	local grad_outputs = self.grad_inputs[self.sequence_step + 1]

	if grad_outputs == nil then
		self.grad_inputs[self.sequence_step + 1] = self:getFirstGradOutputs()
		grad_outputs = self.grad_inputs[self.sequence_step + 1]
	end

	grad_outputs[1] = gradOutput

	local grad_inputs = self.ctrl[self.sequence_step]:backward(inputs, grad_outputs)
	
	self.grad_inputs[self.sequence_step] = grad_inputs

	if self.sequence_step == 1 then
		grad_inputs[1] = torch.Tensor{0}
		self.initilizer:backward(torch.Tensor{0},grad_inputs)
		self.grad_inputs = {}
	end


	self.sequence_step = self.sequence_step - 1
end

function NTM:parameters()
	ctrl_p, ctrl_g = self.ctrl[1]:parameters()
	init_p, init_g = self.initilizer:parameters()

	tablex.insertvalues(ctrl_p, init_p)
	tablex.insertvalues(ctrl_g, init_g)

	return ctrl_p, ctrl_g
end

function NTM:zeroGradParameters()
	self.ctrl[1]:zeroGradParameters()
	self.initilizer:zeroGradParameters()
end

function NTM:new_sequence()
	self.sequence_step = 0
end

