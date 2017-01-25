utils = require 'utils'

tasks = {}

function tasks.create_sample(sample_size, is_flag, flags)

    local res
    if is_flag then
        res = torch.zeros(unpack(sample_size))
    else
        res = torch.rand(unpack(sample_size)):gt(0.5):double()
    end
    res[{{1},{-#flags, -1}}] = torch.Tensor(flags)
    return res
end


function tasks.generate_sequence(n_sample, sample_size, flags)

    local inputs = torch.zeros(n_sample, sample_size)

    for i=1, n_sample do
        inputs[i] = tasks.create_sample({1, sample_size}, false, flags)
    end
    return inputs
end


function tasks.generate_copy_sequence(n_sample, sample_size, force_zero)
    local total_elems =  2 * n_sample + 1

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}


    local seq = tasks.generate_sequence(n_sample, sample_size, {0})
    inputs[{{1, n_sample},{}}] = seq
    inputs[n_sample + 1] = tasks.create_sample({1, sample_size}, true, {1})

    targets[{{n_sample + 2, -1},{}}] = seq

    for i=1,total_elems do
        expect_out[i] = i > n_sample + 1 or force_zero
    end

    return inputs, targets, expect_out
end



function tasks.generate_repeat_copy_sequence(n_sample, sample_size, n_repeat, force_zero)
    local total_elems = n_sample * (n_repeat + 1) + 2

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}

    local seq = tasks.generate_sequence(n_sample, sample_size, {0})
    inputs[{{1, n_sample},{}}] = seq
    inputs[n_sample + 1] = tasks.create_sample({1, sample_size}, true, {1})

    local beg_ind = n_sample + 2
    for i=1,n_repeat do
        targets[{{beg_ind, beg_ind + n_sample -1 },{}}] = seq
        beg_ind = beg_ind + n_sample
    end
    targets[-1] = tasks.create_sample({1, sample_size}, true, {1})

    for i=1,total_elems do
        expect_out[i] = i > n_sample + 1 or force_zero
    end

    return inputs, targets, expect_out
end

function tasks.generate_associative_racall_sequence(sequence_size, item_length, key_item, sample_size, force_zero)
    local total_elems =  (item_length + 1) * (sequence_size + 2)

    local inputs = torch.zeros(total_elems, sample_size)
    local targets = torch.zeros(total_elems, sample_size)
    local expect_out = {}

    local key_ind = sequence_size * (item_length + 1) + 2

    local seq 
    local beg_ind = 2
    for i=1,sequence_size  do
        inputs[{{beg_ind - 1},{}}] = tasks.create_sample({1, sample_size}, true, {1,0})

        seq = tasks.generate_sequence(item_length, sample_size, {0,0})
        inputs[{{beg_ind, beg_ind + item_length - 1},{}}] = seq
        if i == key_item then
            inputs[{{key_ind, key_ind + item_length - 1},{}}] = seq
            inputs[{{key_ind-1},{}}] = tasks.create_sample({1, sample_size}, true, {0,1})
            inputs[{{key_ind + item_length },{}}] = tasks.create_sample({1, sample_size}, true, {0,1})
        elseif i == key_item + 1 then
            targets[{{-item_length,-1}}] = seq
        end

        beg_ind = beg_ind + item_length + 1
    end

    for i=1,total_elems do
        expect_out[i] = i > total_elems - item_length or force_zero
    end

    return inputs, targets, expect_out
end

function tasks.launch_task(task_params, ntm_params, optim_params, seed)
    if seed then
        torch.manualSeed(seed)
    end

    local ntm_model = nn.NTM(ntm_params)
    local criterion = nn.BCECriterion()

    local start_time = os.date('%d.%m.%Y_%X')
    local dir_format = "params/%s/%s_len=%d-%d_lr=%.4f"
    local sep = '-'
    local save_dir = dir_format:format(task_params.task_name, start_time, task_params.min_seq_len, task_params.max_seq_len, optim_params.learningRate)
    os.execute("mkdir -p " .. save_dir)


    local running_error = {}
    local errors = {}
    local model_params, model_grads = ntm_model:getParameters() 
    
    for i=1,task_params.n_epochs do

        local feval = function(x)
            local s = os.clock()
            
            local inputs, targets, expect_out = task_params.data_gen()

            local outs = torch.Tensor(targets:size())
            local total_length = inputs:size(1)
            for j=1,total_length do
                outs[j] = ntm_model:forward(inputs[j])
            end
            
            ntm_model:zeroGradParameters()

            local err = 0

            local real_outs

            local s = os.clock()
            for j=total_length,1,-1 do
                if expect_out[j] then
                    if not real_outs then
                        real_outs = outs[j]:view(1,-1)
                    else
                        real_outs = outs[j]:view(1,-1):cat(real_outs,1)
                    end

                    err = err + criterion:forward(outs[j], targets[j])
                    grad = criterion:backward(outs[j], targets[j])
                    
                else
                    grad = torch.zeros(1,ntm_params.output_size)
                end
    
                ntm_model:backward(inputs[j],grad)

            end
            local t = os.clock()

            n_out = real_outs:size(1)
            err = err / n_out

            local n_sup = model_grads:gt(task_params.clip_max):sum()
            local n_inf = model_grads:lt(task_params.clip_min):sum()

            model_grads:clamp(task_params.clip_min, task_params.clip_max)

            running_error[i % task_params.running_error_size] = err
           
            if i % task_params.running_error_size == 1 then
                table.insert(errors, torch.Tensor(running_error):mean())
                local x = torch.range(0,#errors -1) * task_params.running_error_size
                utils.save_plot({x,torch.Tensor(errors)}, task_params.figure_name:format(save_dir))
            end

            if i % task_params.print_period == 1 then
                local string_sep = '\n%s\nIteration n°%d\t time : %fs\n'
                io.write(string_sep:format(sep:rep(30), i, t-s))

                io.write('\nInputs :\n')
                io.write(tostring(inputs))

                io.write('\nOutputs :\n')
                -- io.write(tostring(real_outs))
                io.write(tostring(outs))

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

            if i % task_params.save_period == 0 then
                local var_name = '%s/%d-%.5f.params'
                torch.save(var_name:format(save_dir,i,torch.Tensor(running_error):mean()), model_params)
            end

            return err, model_grads
        end
        utils.rmsprop(feval, model_params, optim_params)
    end               
end


return tasks