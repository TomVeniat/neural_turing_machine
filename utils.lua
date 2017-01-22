local utils = {}

function utils.createSample(sample_size, start_tag, end_tag)
    local res
    if start_tag or end_tag then
        res = torch.zeros(unpack(sample_size))
    else 
        res = torch.rand(unpack(sample_size)):gt(0.5):double()
    end

    res[1][-2] = start_tag and 1 or 0
    res[1][-1] = end_tag and 1 or 0
    return res
end

function utils.generate_sequence(n_sample, sample_size)

    local inputs = torch.zeros(n_sample, sample_size)

    for i=1, n_sample do
        inputs[i] = utils.createSample({1, sample_size},false, false)
    end
    return inputs
end


function utils.generate_copy_sequence(n_sample, sample_size)
    local begin_flag = utils.createSample({1, sample_size},true, false)
    local end_flag = utils.createSample({1, sample_size},false, true)
    local zeros = torch.zeros(1, sample_size)

    local inputs = torch.zeros(2 * n_sample + 2, sample_size)

    inputs[1] = begin_flag
    inputs[n_sample + 2] = end_flag
    for j=1,n_sample do
        inputs[j + 1] = utils.createSample({1, sample_size},false, false)
    end
    return inputs
end

function utils.generate_repeat_copy_sequence(n_sample, sample_size, n_repeat)
    local total_elems = n_sample * (n_repeat + 1) + 3

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}

    local seq = utils.generate_sequence(n_sample, sample_size)
    inputs[1] = utils.createSample({1, sample_size},true, false)
    inputs[n_sample + 2] = utils.createSample({1, sample_size},false, true)
    inputs[{{2, n_sample + 1},{}}] = seq

    local beg_ind = n_sample + 3
    for i=1,n_repeat do
        targets[{{beg_ind, beg_ind + n_sample -1 },{}}] = seq
        beg_ind = beg_ind + n_sample
    end
    targets[-1] = utils.createSample({1, sample_size},false, true)

    for i=1,total_elems do
        expect_out[i] = i > n_sample+2
    end


    return inputs, targets, expect_out
end


--[[

Rms prop as described in by Graves in http://arxiv.org/pdf/1308.0850v5.pdf, Sec 4.2

Implementation by Kai Sheng Tai (https://github.com/kaishengtai/torch-ntm)

--]]
function utils.rmsprop(opfunc, x, config, state)
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


function utils.save_plot(data, file_name)
    gnuplot.pngfigure(file_name)
    gnuplot.plot(data)
    gnuplot.plotflush()
end

return utils