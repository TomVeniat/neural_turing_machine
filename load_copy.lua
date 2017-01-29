require 'nn'
require 'ntm'
require 'gnuplot'
local tasks = require 'tasks'

local ntm_params = {
    input_size = 5,
    output_size = 5,
    mem_locations = 10,
    mem_location_size = 20,
    hidden_state_size = 100,
    allowed_shifts = {-1,0,1}
}

local ntm = nn.NTM(ntm_params)

-- Parameters of a model trained with sequences of length 25.
-- local loaded_params = torch.load('parameters/copy_force/24.01.2017_14:26:31_len=1-20_lr=0.0001/25000-0.00001.params')
-- local loaded_params = torch.load('parameters/copy_no_force/24.01.2017_12:32:00_len=1-20_lr=0.0001/25000-0.00002.params')
-- Smaller model :
local loaded_params = torch.load('parameters/copy/toy/copy_force=false_seed=1/29.01.2017_10:58:15_len=1-7_lr=0.0001/25000-0.00002.params')


local ntm_p, ntm_g = ntm:getParameters()
ntm_p:copy(loaded_params)


local min_seq_len = 12
local max_seq_len = 12

local crit = nn.BCECriterion()

io.write('Sequence length\t\tError\n')
for i = min_seq_len, max_seq_len do
    local seq_len = i
    local inputs, targets, exp_out  = tasks.generate_copy_sequence(seq_len, ntm_params.input_size, false)

    local out = torch.Tensor(targets:size())

    local err = 0
    local n_out = 0
    for j=1,inputs:size(1) do
        out[j] = ntm:forward(inputs[j])
        if exp_out[j] then
            err = err + crit:forward(out[j], targets[j])
            n_out = n_out + 1
        end
    end

    err = err / n_out
    print(inputs)

    print(out)

    local diff = (targets - out):abs() 
    print (diff)

    io.write('\nStep\tWrites\t\t\tReads\n')
    for j=1,inputs:size(1) do
        local vals_w, inds_w = ntm.outputs[j][3]:sort(true)
        local vals_r, inds_r = ntm.outputs[j][5]:sort(true)
        local str = '%d\t%d\t%.3f\t\t%d\t%.3f\n'
        io.write(str:format(j,inds_w[1], vals_w[1], inds_r[1], vals_r[1]))
    end


    gnuplot.figure(1)
    gnuplot.imagesc(inputs:t(),'jet')

    gnuplot.figure(2)
    gnuplot.imagesc(out:t(),'color')

    gnuplot.figure(3)
    gnuplot.imagesc(diff:t(),'color')


    local str_format = '%d\t\t\t%f\n'
    io.write(str_format:format(seq_len, err))
    io.flush()
    ntm:new_sequence()
end
