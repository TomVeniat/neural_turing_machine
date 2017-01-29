require 'nn'
require 'ntm'
require 'gnuplot'

local tasks = require 'tasks'

local sim_params = {
	--Total sequence number used for training
	n_epochs = 25000,
	--Training sequence length bounds
	min_seq_len = 1,
	max_seq_len = 7,
	--Debug output period (-1 for no console output)
	print_period = 100,
	--Params saving period (-1 for no parameters backups)
	save_period = 1000,
	--"Batch" size, number of sequence processed between two plot points (-1 for no plot)
	running_error_size = 100,
	figure_name = '%s/loss.svg',
	--Gradient clipping bounds
	clip_max = 10,
	clip_min = -10,
	--force ntm to output zeros when no output expected
	force_zero = false
}

local ntm_params = {
    input_size = 5,
    output_size = 5,
    mem_locations = 10,
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
    return tasks.generate_copy_sequence(seq_len, ntm_params.input_size, sim_params.force_zero)
end

local n_repeat = 5
function data_gen_rep_copy()
	local seq_len = torch.random(sim_params.min_seq_len, sim_params.max_seq_len)
    return tasks.generate_repeat_copy_sequence(seq_len, ntm_params.input_size, n_repeat, sim_params.force_zero)
end


local item_length = 3
function data_gen_assoc_recall()
	local seq_len = torch.random(sim_params.min_seq_len, sim_params.max_seq_len)
	local key_index = torch.random(1, seq_len - 1)

	return tasks.generate_associative_racall_sequence(seq_len, item_length, key_index, ntm_params.input_size, sim_params.force_zero)
end

sim_params.data_gen = data_gen_copy

local test_force = {false}

for i=1,1 do
	for k,v in ipairs(test_force) do
		local task_name = 'copy/toy/copy_force='..tostring(v)..'_seed='..i

		sim_params.force_zero = v
		sim_params.task_name = task_name
		tasks.launch_task(sim_params, ntm_params, rmsprop_config, i)
	end
end

