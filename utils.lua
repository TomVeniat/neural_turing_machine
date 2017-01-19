
function utils.createSample(sampleSize, start_tag, end_tag)
    local res
    if start_tag or end_tag then
        res = torch.zeros(unpack(sampleSize))
    else 
        res = torch.rand(unpack(sampleSize)):gt(0.5):double()
    end

    res[1][-2] = start_tag and 1 or 0
    res[1][-1] = end_tag and 1 or 0
    return res
end

function utils.generate_sequence(nSample, sampleSize)
    local begin_flag = utils.createSample({1, sampleSize},true, false)
    local end_flag = utils.createSample({1, sampleSize},false, true)
    local zeros = torch.zeros(1, sampleSize)

    local inputs = torch.Tensor(2 * nSample + 2, sampleSize)

    inputs[1] = begin_flag
    for j=1,nSample do
        inputs[j + 1] = utils.createSample({1, sampleSize},false, false)
        inputs[nSample + 2 + j ] = zeros
    end
    inputs[nSample + 2] = end_flag
    return inputs
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