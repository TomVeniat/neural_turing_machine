require 'nn'
require 'ntm'
require 'gnuplot'
local tasks = require 'tasks'

local ntm_params = {
    input_size = 7,
    output_size = 7,
    mem_locations = 128,
    mem_location_size = 20,
    hidden_state_size = 100,
    allowed_shifts = {-1,0,1}
}

local ntm = nn.NTM(ntm_params)

-- Trained with zeros as targets for input phase :
-- input size : 7, output size : 7, n memeory slots : 128
local loaded_params = torch.load('parameters/assoc_recall/25000-0.00001.params')


local ntm_p, ntm_g = ntm:getParameters()
ntm_p:copy(loaded_params)


local min_seq_len = 15
local max_seq_len = 15

local crit = nn.BCECriterion()

local item_length = 3
for i = min_seq_len, max_seq_len do
    local seq_len = i
    local key_index = torch.random(1, seq_len - 1)
    local inputs, targets, exp_out  = tasks.generate_associative_racall_sequence(seq_len, item_length, key_index, ntm_params.input_size, true)

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
    io.write('Input : \n')
    print(inputs)

    io.write('Output : \n')
    print(out)

    local diff = (targets - out):abs() 
    io.write('Error : \n')
    print (diff)

    gnuplot.figure(1)
    gnuplot.imagesc(inputs:t(),'color')

    gnuplot.figure(2)
    gnuplot.imagesc(out:t(),'color')

    gnuplot.figure(3)
    gnuplot.imagesc(diff:t(),'color')

    io.write('Sequence length\t\tError\n')
    local str_format = '%d\t\t\t%f\n'
    io.write(str_format:format(seq_len, err))
    io.flush()
    ntm:new_sequence()
end
