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
    local total_elems =  2 * n_sample + 2

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}


    local seq = utils.generate_sequence(n_sample, sample_size)
    inputs[1] = utils.createSample({1, sample_size},true, false)
    inputs[n_sample + 2] = utils.createSample({1, sample_size},false, true)
    inputs[{{2, n_sample + 1},{}}] = seq

    targets[{{n_sample + 3, total_elems},{}}] = seq

    for i=1,total_elems do
        expect_out[i] = i > n_sample + 2
    end
    return inputs, targets, expect_out
end



function utils.generate_repeat_copy_sequence(n_sample, sample_size, n_repeat)
    local n_repeat = n_repeat or 3
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
        expect_out[i] = i > n_sample + 2
    end

    return inputs, targets, expect_out
end

function utils.generate_associative_racall_sequence(sequence_size, item_length, key_item, sample_size)
    local total_elems =  (item_length + 1) * (sequence_size + 2)

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}

    local key_ind = sequence_size * (item_length + 1) + 2

    local seq 
    local beg_ind = 2
    for i=1,sequence_size  do
        inputs[{{beg_ind - 1},{}}] = utils.createSample({1, sample_size},true, false)

        seq = utils.generate_sequence(item_length, sample_size)
        inputs[{{beg_ind, beg_ind + item_length - 1},{}}] = seq
        if i == key_item then
            inputs[{{key_ind, key_ind + item_length - 1},{}}] = seq
            inputs[{{key_ind-1},{}}] = utils.createSample({1, sample_size}, false, true)
            inputs[{{key_ind + item_length },{}}] = utils.createSample({1, sample_size}, false, true)
        elseif i == key_item + 1 then
            targets[{{-item_length,-1}}] = seq
        end

        beg_ind = beg_ind + item_length + 1
    end

    for i=1,total_elems do
        expect_out[i] = i > total_elems - item_length
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

function utils.launch_task(task_params, ntm_params, optim_params, seed)
    if seed then
        torch.manualSeed(seed)
    end

    local ntm_model = nn.NTM(ntm_params)
    local criterion = nn.BCECriterion()

    local start_time = os.date('%d.%m.%Y_%X')
    local dir_format = "params/%s_len=%d-%d_lr=%.4f"
    local sep = '-'
    local save_dir = dir_format:format(start_time, task_params.min_seq_len, task_params.max_seq_len, optim_params.learningRate)
    os.execute("mkdir -p " .. save_dir)


    local running_error = {}
    local errors = {}
    local model_params, model_grads = ntm_model:getParameters() 
    
    for i=1,task_params.n_epochs do

        local feval = function(x)
            
            local seq_len, inputs, targets, expect_out = task_params.data_gen()

            local outs = torch.Tensor(inputs:size())
            local total_length = inputs:size(1)
            for j=1,total_length do
                outs[j] = ntm_model:forward(inputs[j])
            end
            ntm_model:zeroGradParameters()
            local err = 0
            for j=total_length,1,-1 do
                local grad
                if expect_out[j] then
                    err = err + criterion:forward(outs[j], targets[j])
                    grad = criterion:backward(outs[j], targets[j])
                else
                    grad = torch.zeros(1, ntm_params.output_size)
                end
                ntm_model:backward(inputs[j],grad)
            end
            local n_sup = model_grads:gt(task_params.clip_max):sum()
            local n_inf = model_grads:lt(task_params.clip_min):sum()

            model_grads:clamp(task_params.clip_min, task_params.clip_max)

            running_error[i % task_params.running_error_size] = err
           
            if i % task_params.running_error_size == 1 then
                table.insert(errors, torch.Tensor(running_error):mean())
                utils.save_plot(torch.Tensor(errors), task_params.figure_name:format(save_dir))
            end

            if i % task_params.print_period == 1 then
                local string_sep = '\n%s\nIteration n°%d, sequence length : %d\n'
                io.write(string_sep:format(sep:rep(30), i, seq_len))

                io.write('\nInputs :\n')
                io.write(tostring(inputs[{{1,seq_len}}]))

                io.write('\nOutputs :\n')
                io.write(tostring(outs[{{seq_len + 1,-1}}]))

                io.write('\nStep\tWrites\t\t\tReads\n')
                for j=1,total_length do
                    local vals_w, inds_w = ntm_model.outputs[j][3]:sort(true)
                    local vals_r, inds_r = ntm_model.outputs[j][5]:sort(true)
                    local str = '%d\t%d\t%.3f\t\t%d\t%.3f\n'
                    io.write(str:format(j,inds_w[1], vals_w[1], inds_r[1], vals_r[1]))
                end

                local errors_log = '\nError on last sequence : \t %.5f\nError on last %d sequences : \t %.5f\n'
                io.write(errors_log:format(err, task_params.running_error_size, torch.Tensor(running_error):mean()))
                io.write('\nGradients clipped : ', n_inf + n_sup, '\n')
                io.flush()
            end

            if i % task_params.save_period == 1 then
                local var_name = '%s/%d-%.5f.params'
                torch.save(var_name:format(save_dir,i,torch.Tensor(running_error):mean()), model_params)
            end

            return err, model_grads
        end
        utils.rmsprop(feval, model_params, optim_params)
    end               
end

return utils
