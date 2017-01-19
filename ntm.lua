require 'nn'
require 'nngraph'
require 'modules/MulScalar'
require 'modules/Shifter'
require 'modules/PowScalar'

local NTM, Parent = torch.class('nn.NTM', 'nn.Module')

function NTM:__init( params)
    Parent.__init(self)

    self.input_size = params.input_size or 10
    self.output_size = params.output_size or 10
    self.mem_locations = params.mem_locations or 120
    self.mem_location_size = params.mem_location_size or 20
    self.hidden_state_size = params.hidden_state_size or 100
    self.allowed_shifts = params.allowed_shifts or {-1,0,1}

    self.ctrl = {self:new_unit()}
    
    self.initilizer = self:get_initializer()

    self.inputs = {}
    self.outputs = {}

    self.sequence_step = 0

end

function NTM:new_unit()
    local input = nn.Identity()()

    local prev_mem = nn.Identity()()
    local prev_read_w = nn.Identity()()
    local prev_read = nn.Identity()()
    local prev_write_w = nn.Identity()()

    local in_hidden = nn.Linear(self.input_size, self.hidden_state_size)(input)
    local r_hidden = nn.Linear(self.mem_location_size, self.hidden_state_size)(prev_read)
    local controller = nn.CAddTable()({in_hidden, r_hidden})

    local read, read_w = self:create_read_head(controller, prev_read_w, prev_mem)
    local memory, write_w = self:create_write_head(controller, prev_write_w, prev_mem)

    local output = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.output_size)(controller))

    local inputs = {input, prev_mem, prev_write_w, prev_read, prev_read_w}
    local outputs = {output, memory, write_w, read, read_w}

    local new_module = nn.gModule(inputs, outputs)

    return new_module
end

function NTM:copy_main_unit()
    local new_unit = self:new_unit()

    local params, grads = self.ctrl[1]:parameters()
    local new_params, new_grads = new_unit:parameters()

    for i = 1, #params do
        new_params[i]:set(params[i])
        new_grads[i]:set(grads[i])
    end

    return new_unit
end

function NTM:get_initializer()
    local input = nn.Identity()()

    local mem = nn.View(self.mem_locations,self.mem_location_size)(nn.Linear(1,self.mem_locations * self.mem_location_size)(input))
    local wr_lin = nn.Linear(1,self.mem_locations)
    wr_lin.bias:copy(torch.range(self.mem_locations, 1, -1))
    local wr = nn.SoftMax()(wr_lin(input))

    local ww_lin = nn.Linear(1,self.mem_locations)
    ww_lin.bias:copy(torch.range(self.mem_locations, 1, -1))
    local ww = nn.SoftMax()(ww_lin(input))
    local r = nn.Tanh()(nn.Linear(1,self.mem_location_size)(input))

    return nn.gModule({input},{nn.Identity()(input),mem,ww,r,wr})
end

function NTM:get_first_grad_outputs()
    local mem_grad = torch.zeros(self.mem_locations, self.mem_location_size)
    local ww_grad = torch.zeros(self.mem_locations)
    local r_grad = torch.zeros(self.mem_location_size)
    local wr_grad = torch.zeros(self.mem_locations)
    return {0, mem_grad, ww_grad, r_grad, wr_grad}

end

function NTM:create_read_head(ctrl_state, prev_w, mem)
    local w = self:create_head(ctrl_state, prev_w, mem)
    local read = nn.MixtureTable()({w, mem})
    return read, w
end

function NTM:create_write_head(ctrl_state, prev_w, mem)
    local w = self:create_head(ctrl_state, prev_w, mem)

    local e = nn.Sigmoid()(nn.Linear(self.hidden_state_size, self.mem_location_size)(ctrl_state))
    local erasure = nn.AddConstant(1)(
                        nn.MulConstant(-1)(
                            nn.MM()({
                                nn.View(self.mem_locations,1)(w),
                                nn.View(1, self.mem_location_size)(e)})))

    local a = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(ctrl_state))
    local addition = nn.MM()({
      nn.View(self.mem_locations,1)(w),
      nn.View(1, self.mem_location_size)(a)})

    local erased_mem = nn.CMulTable()({mem, erasure})
    local new_mem = nn.CAddTable()({erased_mem,addition})

    return new_mem, w
end

function NTM:create_head(ctrl_state, prev_w, mem)
    local key = nn.Tanh()(nn.Linear(self.hidden_state_size, self.mem_location_size)(ctrl_state))
    local beta = nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(ctrl_state))
    local g = nn.Sigmoid()(nn.Linear(self.hidden_state_size, 1)(ctrl_state))
    local shift_weigths = nn.SoftMax()(nn.Linear(self.hidden_state_size,#self.allowed_shifts)(ctrl_state))
    local gamma = nn.AddConstant(1)(nn.SoftPlus()(nn.Linear(self.hidden_state_size, 1)(ctrl_state)))

    local key_matching = nn.CosineDistance()({mem, nn.Replicate(self.mem_locations)(key)})

    local w_c = nn.SoftMax()(nn.MulScalar()({beta, key_matching}))

    local w_g1 = nn.MulScalar()({g, w_c})
    local w_g2 = nn.MulScalar()({nn.AddConstant(1)(nn.MulConstant(-1)(g)), prev_w})
    local w_g = nn.CAddTable()({w_g1,w_g2})

    local w_s = nn.Shifter(self.allowed_shifts)({w_g, shift_weigths})

    local w = nn.Normalize(1)(nn.PowScalar()({gamma, w_s}))

    return w
end


function NTM:forward(input)
    self.sequence_step = self.sequence_step + 1

    local current_unit = self.ctrl[self.sequence_step]
    if current_unit == nil then
       current_unit = self:copy_main_unit()
       table.insert(self.ctrl, current_unit)
    end

    local inputs
    if self.sequence_step == 1 then
       inputs = self.initilizer:forward(torch.Tensor{0})
    else
       inputs = self.outputs[self.sequence_step - 1]
    end
    inputs[1] = input

    self.inputs[self.sequence_step] = inputs
    self.outputs[self.sequence_step] = current_unit:forward(inputs)

    return self.outputs[self.sequence_step][1]
end

function NTM:backward(input, gradOutput)
    
    local inputs = self.inputs[self.sequence_step]

    if self.grad_inputs == nil then
       self.grad_inputs = self:get_first_grad_outputs()
    end
    self.grad_inputs[1] = gradOutput

    self.grad_inputs = self.ctrl[self.sequence_step]:backward(inputs, self.grad_inputs)

    if self.sequence_step == 1 then
       self.grad_inputs[1] = torch.Tensor{0}
       self.initilizer:backward(torch.Tensor{0},self.grad_inputs)
       self.grad_inputs = nil
    end

    self.sequence_step = self.sequence_step - 1
end

function NTM:parameters()
    local ctrl_params, ctrl_grads = self.ctrl[1]:parameters()
    local init_params, init_grads = self.initilizer:parameters()

    local flattener = nn.FlattenTable()

    local params = flattener:forward({ctrl_params,init_params})
    local grads = flattener:forward({ctrl_grads,init_grads})

    return params, grads
end

function NTM:zeroGradParameters()
    self.ctrl[1]:zeroGradParameters()
    self.initilizer:zeroGradParameters()
end

function NTM:new_sequence()
    self.sequence_step = 0
    self.grad_inputs = nil
    self.inputs = {}
    self.outputs = {}
end
