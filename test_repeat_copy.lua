require 'nn'
require 'ntm'
require 'gnuplot'

local utils = require 'utils'

local sep = '-'

local t = 10e6
local min_seq_len = 1
local max_seq_len = 7
local print_period = 100
local save_period = 1000
local running_error_size = 500
local figure_name = '%s/loss.png'

local ntm_params = {
    input_size = 5,
    output_size = 5,
    mem_locations = 128,
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

    local n_repeat = 3

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
            local inputs, targets, expect_out = utils.generate_repeat_copy_sequence(seq_len, ntm_params.input_size, n_repeat)

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
                io.write(tostring(outs[{{seq_len+3,-1}}]))

                io.write('\nStep\tWrites\t\t\tReads\n')
                for j=1,total_length do
                    local vals_w, inds_w = ntm_model.outputs[j][3]:sort(true)
                    local vals_r, inds_r = ntm_model.outputs[j][5]:sort(true)
                    local str = '%d\t%d\t%.3f\t\t%d\t%.3f\n'
                    io.write(str:format(j,inds_w[1], vals_w[1], inds_r[1], vals_r[1]))
                end

                local errors_log = '\nError on last sequence : \t %.5f\nError on last %d sequences : \t %.5f\n'
                io.write(errors_log:format(err, running_error_size, torch.Tensor(running_error):mean()))
                io.write('\nGradients clipped : ', n_inf + n_sup, '\n')
                io.flush()
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
