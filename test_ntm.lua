require 'nn'
require 'ntm'
require 'gnuplot'

local utils = require 'utils'

local sim_params = {
	n_epochs = 10e6,
	min_seq_len = 3,
	max_seq_len = 3,
	print_period = 100,
	save_period = 1000,
	running_error_size = 500,
	figure_name = '%s/loss.png',
	clip_max = 10,
	clip_min = -10
}

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


function data_gen_copy()
	local seq_len = torch.random(sim_params.min_seq_len, sim_params.max_seq_len)
    return seq_len + 2, utils.generate_copy_sequence(seq_len, ntm_params.input_size)
end

local n_repeat = 5
function data_gen_rep_copy()
	local seq_len = torch.random(sim_params.min_seq_len, sim_params.max_seq_len)
    return seq_len + 2, utils.generate_repeat_copy_sequence(seq_len, ntm_params.input_size, n_repeat)
end


local item_length = 3
function data_gen_assoc_recall()
	local seq_len = torch.random(sim_params.min_seq_len, sim_params.max_seq_len)
	local key_index = torch.random(1, seq_len - 1)

	in_len = (seq_len + 1) * (item_length + 1)
	return in_len, utils.generate_associative_racall_sequence(seq_len, item_length, key_index, ntm_params.input_size)
end

sim_params.data_gen = data_gen_assoc_recall


utils.launch_task(sim_params, ntm_params, rmsprop_config, 12)
