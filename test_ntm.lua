require 'nn'
require 'ntm'
require 'gnuplot'
require 'utils'

function createSample(sampleSize, start_tag, end_tag)
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

function generate_sequence(nSample, sampleSize)
    local begin_flag = createSample({1, sampleSize},true, false)
    local end_flag = createSample({1, sampleSize},false, true)
    local zeros = torch.zeros(1, sampleSize)

    local inputs = torch.Tensor(2 * nSample + 2, sampleSize)

    inputs[1] = begin_flag
    for j=1,nSample do
        inputs[j + 1] = createSample({1, sampleSize},false, false)
        inputs[nSample + 2 + j ] = zeros
    end
    inputs[nSample + 2] = end_flag
    return inputs
end

local sep = '-'

local t = 10000
local min_seq_len = 1
local max_seq_len = 20
local print_period = 100
local save_period = 1000
local running_error_size = 500
local figure_name = '%s/loss.png'

local ntm_params = {
    input_size = 10,
    output_size = 10,
    mem_locations = 50,
    mem_location_size = 20,
    hidden_state_size = 100,
    allowed_shifts = {-1,0,1}
}

local rmsprop_config = {
  learningRate = 1e-4,
  momentum = 0.9,
  decay = 0.95
}

local clip_max = 10
local clip_min = -10

local criterion = nn.BCECriterion()

function launch_copy(seed)
    if seed then
        torch.manualSeed(seed)
    end

    local ntm_model = nn.NTM(ntm_params)

    local start_time = os.date('%d.%m.%Y_%X')
    local dir_format = "params/%s_len=%d-%d_lr=%.4f"
    local save_dir = dir_format:format(start_time, min_seq_len, max_seq_len, rmsprop_config.learningRate)
    os.execute("mkdir -p " .. save_dir)


    local running_error = {}
    local errors = {}
    local model_params, model_grads = ntm_model:getParameters() 
    
    for i=1,t do
        local feval = function(x)
            local seq_len = torch.random(min_seq_len, max_seq_len)
            local inputs = generate_sequence(seq_len, ntm_params.input_size)

            for j=1,seq_len + 2 do
                ntm_model:forward(inputs[j])
            end
            
            outputs = torch.Tensor(seq_len,ntm_params.output_size)
            for j=1,seq_len do
                outputs[j] = ntm_model:forward(inputs[seq_len + 2 + j ])
            end

            ntm_model:zeroGradParameters()

            local grads = torch.Tensor(2 * seq_len + 2, ntm_params.input_size)
            local zeros = torch.zeros(1, ntm_params.input_size)

            grads[{{1,seq_len + 2}}] = zeros:repeatTensor(seq_len + 2, 1)

            local err = 0
            for j=1,seq_len do
                grads[j + seq_len + 2] = criterion:backward(outputs[j], inputs[j + 1])
                err = err + criterion:forward(outputs[j], inputs[j + 1])
            end

            for j = 2 * seq_len + 2, 1, -1 do
                ntm_model:backward(inputs[j],grads[j])
            end
            
            local n_sup = model_grads:gt(clip_max):sum()
            local n_inf = model_grads:lt(clip_min):sum()

            model_grads:clamp(clip_min, clip_max)

            running_error[i % running_error_size] = err

            if i % running_error_size == 1 then
                table.insert(errors, torch.Tensor(running_error):mean())
                utils.save_plot(torch.Tensor(errors), figure_name:format(save_dir))
            end

            if i % print_period == 1 then
                local string_sep = '\n%s\nIteration n°%d, sequence length : %d\n'
                io.write(string_sep:format(sep:rep(30), i, seq_len))

                io.write('\nInputs :\n')
                io.write(tostring(inputs[{{2,seq_len+1}}]))

                io.write('\nOutputs :\n')
                io.write(tostring(outputs))

                io.write('\nStep\tWrites\t\t\tReads\n')
                for j=1,2 * seq_len + 2 do
                    local vals_w, inds_w = ntm_model.outputs[j][3]:sort(true)
                    local vals_r, inds_r = ntm_model.outputs[j][5]:sort(true)
                    local str = '%d\t%d\t%.3f\t\t%d\t%.3f\n'
                    io.write(str:format(j,inds_w[1], vals_w[1], inds_r[1], vals_r[1]))
                end

                local errors_log = '\nError on last sequence : \t %.5f\nError on last %d sequences : \t %.5f\n'
                io.write(errors_log:format(err, running_error_size, torch.Tensor(running_error):mean()))
                io.write('\nGradients clipped : ', n_inf + n_sup, '\n')
            end

            if i % save_period == 1 then
                local var_name = '%s/%d-%.5f.params'
                torch.save(var_name:format(save_dir,i,torch.Tensor(running_error):mean()), model_params)
            end

            return err, model_grads
        end
        utils.rmsprop(feval, model_params, rmsprop_config)
    end               
end

launch_copy(12)
