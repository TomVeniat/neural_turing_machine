local Shifter, _ = torch.class('nn.Shifter', 'nn.Module')


function Shifter:__init(shift_range)
	self.shift_range = shift_range
end

function rotate_vector(v, s)
	local size
	if v:dim() == 1 then
		size = v:size(1)
	else
		size = v:size(2)
		-- error('Not supported')
	end
	local v_copy = v:repeatTensor(1,3)
	return v_copy[{{1},{size + 1 - s, 2 * size - s}}]:view(v:size())
end

function Shifter:buildShiftMat(shifts, size)
	local shiftMat = torch.zeros(size,size)

	for i=0,size-1 do
		for j=1,#self.shift_range  do
			local s = self.shift_range[j]
			local val = shifts[j]
			local index = math.fmod(i - s + size, size) + 1
			shiftMat[{{index},{i+1}}] = val
		end
	end
	
	return shiftMat
end


function Shifter:updateOutput(input)
	local weights = input[1]
	local shifts = input[2]

	if shifts:dim() == 2 then
		shifts = shifts:view(shifts:size(2))
	end

	local size
	if weights:dim() == 2 then
		size = weights:size(2)
	else
		size = weights:size(1)
	end

	shiftMat = self:buildShiftMat(shifts,size)

	self.output = (weights:view(1,size) * shiftMat):view(weights:size())

	return self.output
end


function Shifter:updateGradInput(input, gradOutput)
	local weights = input[1]
	local shifts = input[2]	
	local size
	if weights:dim() == 2 then
		size = weights:size(2)
	else
		size = weights:size(1)
	end

	if shifts:dim() == 2 then
		shifts = shifts:view(shifts:size(2))
	end

	local shiftMat = self:buildShiftMat(shifts,size)

	self.gradInput = {}
	self.gradInput[1] = shiftMat * gradOutput:view(size,1)

	local grad_shifts = torch.Tensor():resize(shifts:size())
	for i=1,#self.shift_range do
		shifted_weights = rotate_vector(weights, self.shift_range[i])
		grad_shifts[i] = gradOutput:dot(shifted_weights)
	end

	self.gradInput[2] = grad_shifts:view(shifts:size())

	return self.gradInput
end