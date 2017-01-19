require 'nn'
require 'ntm'
local utils = require 'utils'

local ntm_params = {
    input_size = 10,
    output_size = 10,
    mem_locations = 120,
    mem_location_size = 20,
    hidden_state_size = 100,
    allowed_shifts = {-1,0,1}
}

local ntm = nn.NTM(ntm_params)

-- Parameters of a model trained with sequences of length 25.
local loaded_params = torch.load('params/18.01.2017_04:05:51_len=25_lr=0.0001/35000-0.00010.params')

local ntm_p, ntm_g = ntm:getParameters()
ntm_p:copy(loaded_params)

local begin_flag = utils.createSample({1,ntm_params.input_size},true, false)
local end_flag = utils.createSample({1,ntm_params.input_size},false, true)
local zeros = torch.zeros(1,ntm_params.input_size)

local min_seq_len = 115
local max_seq_len = 125

local crit = nn.BCECriterion()

io.write('Sequence length\t\tError\n')
for i = min_seq_len, max_seq_len do
    local seq_len = i
    local seq = utils.generate_sequence(seq_len, 10)

    local out = torch.Tensor(2 * seq_len + 2, ntm_params.input_size)

    local err = 0
    for j=1,seq:size(1) do
        out[j] = ntm:forward(seq[j])
    end

    local str_format = '%d\t\t\t%f\n'
    io.write(str_format:format(seq_len, crit:forward(out[{{seq_len + 3, 2 * seq_len + 2}}],seq[{{2, seq_len + 1 }}])))
    ntm:new_sequence()
end
