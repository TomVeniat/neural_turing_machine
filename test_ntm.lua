require 'nn'
require 'nngraph'
require 'ntm'
require 'optim'

function rmsprop(opfunc, x, config, state)
  if config == nil and state == nil then
    error('no state provided')
  end
  local config = config or {}
  local state = state or config
  local decay = config.decay or 0.95
  local lr = config.learningRate or 1e-4
  local momentum = config.momentum or 0.9
  local epsilon = config.epsilon or 1e-4
  state.evalCounter = state.evalCounter or 0
  state.gradAccum = state.gradAccum or torch.Tensor():typeAs(x):resizeAs(x):zero()
  state.gradSqAccum = state.gradSqAccum or torch.Tensor():typeAs(x):resizeAs(x):zero()
  state.update = state.update or torch.Tensor():typeAs(x):resizeAs(x):zero()
  state.gradRms = state.gradRms or torch.Tensor():typeAs(x):resizeAs(x)

  -- evaluate f(x) and df(x)/dx
  local fx, dfdx = opfunc(x)

  -- accumulate gradients and squared gradients
  -- g_t = d * g_{t-1} + (1 - d) (df/dx)
  state.gradAccum:mul(decay):add(1 - decay, dfdx)
  -- n_t = d * n_{t-1} + (1 - d) (df/dx)^2
  state.gradSqAccum:mul(decay):add(1 - decay, torch.cmul(dfdx, dfdx))

  -- g_t = d * g_{t-1} + (df/dx)
  --state.gradAccum:mul(decay):add(dfdx)
  -- n_t = d * n_{t-1} + (df/dx)^2
  --state.gradSqAccum:mul(decay):add(torch.cmul(dfdx, dfdx))

  -- compute RMS
  -- r_t = \sqrt{ n_t - g_t^2 + \eps }
  state.gradRms:copy(state.gradSqAccum)
    :add(-1, torch.cmul(state.gradAccum, state.gradAccum))
    :add(epsilon)
    :sqrt()

  -- compute update
  -- \Delta_t = m * \Delta_{t-1} - \eta * (df/dx) / r_t
  state.update:mul(momentum):add(-lr, torch.cdiv(dfdx, state.gradRms))

  -- apply update
  x:add(state.update)

  -- (7) update evaluation counter
  state.evalCounter = state.evalCounter + 1

  -- return x*, f(x) before optimization
  return x, {fx}
end


function createSample(sampleSize, start_tag, end_tag)
	-- local result = torch.Tensor():rand(sampleSize):gt(0.5):double()
	local res
	if start_tag or end_tag then
		res = torch.zeros(unpack(sampleSize))
	else 
		res = torch.Tensor():rand(unpack(sampleSize)):gt(0.5):double()
		-- return torch.zeros(unpack(sampleSize))
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



torch.manualSeed(123)

local sep = '-'

local params = {
	input_size = 5,
	output_size = 5,
	mem_locations = 4,
	mem_location_size = 10,
	hidden_state_size = 100,
	allowed_shifts = {-1,0,1}
}

nt = nn.NTM(params)


-- nngraph.annotateNodes()

-- xs, ys = createCopyDataSet(50,3)

local t = 500000
local seq_len = 2
local print_period = 10
local save_period = 10000
local start_time = os.date('%d.%m.%Y_%X')
local save_dir = 'params/' .. start_time
os.execute("mkdir -p " .. save_dir)

local criterion = nn.BCECriterion()

params, grads = nt:getParameters() 
print (params:mean())
print (grads:mean())

local rmsprop_state = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

local begin_flag = createSample({1,5},true, false)
local end_flag = createSample({1,5},false, true)

for i=1,t do
	local stop_flag = false
	local feval = function(x)
		local inputs = {}

		nt:forward(begin_flag)
		for j=1,seq_len do
			inputs[j] = createSample({1,5},false, false)
			nt:forward(inputs[j])
		end

		nt:forward(end_flag)
		
		local zeros = torch.zeros(1,5)
		out = {}
		for j=1,seq_len do
			out[j] = nt:forward(zeros)
		end

		nt:zeroGradParameters()

		err = 0
		for j=seq_len,1,-1 do
			err = err + criterion:forward(out[j], inputs[j])
			grad = criterion:backward(out[j], inputs[j])
			nt:backward(zeros,grad)
		end

		nt:backward(end_flag,zeros)

		for j=seq_len,1,-1 do
			nt:backward(inputs[j],zeros)
		end

		nt:backward(begin_flag,zeros)
 		
		grads:clamp(-10,10)

		if i % print_period == 0 then
			local string_sep = '\n%s\nIteration n°%d:\n'
			io.write(string_sep:format(sep:rep(30),i))

			for j=1,seq_len do
				print(inputs[j])
			end

			for j=1,seq_len do
				print(out[j])
				local vals_w, inds_w = nt.outputs[1+j][3]:sort()
				local vals_r, inds_r = nt.outputs[seq_len + 2 + j][5]:sort()
				local str = '%d\t%.3f\t\t%d\t%.3f\n'
				for k=1,vals_w:size(1) do
					io.write(str:format(inds_r[k], vals_r[k], inds_w[k], vals_w[k]))
				end
			end
			io.write(err,'\n')
			io.write(grads:max(),'\n')
			io.write(grads:min(),'\n')
		end
		if i % save_period == 0 then
			local var_name = '%s/%d.params'
			torch.save(var_name:format(save_dir,i), params)
		end
		return err, grads
	end
	rmsprop(feval, params, rmsprop_state)
end
