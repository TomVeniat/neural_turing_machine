local MulScalar, _ = torch.class('nn.MulScalar', 'nn.Module')

function MulScalar:buildShiftMat(shifts, size)
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


function MulScalar:updateOutput(input)
	local vec = input[1]
	local scal = input[2]

	self.output = vec * scal[1]

	return self.output
end


function MulScalar:updateGradInput(input, gradOutput)
	local vec = input[1]
	local scal = input[2]

	self.gradInput = {}
	self.gradInput[1] = gradOutput * scal[1]
	self.gradInput[2] = torch.Tensor({vec:dot(gradOutput)})

	return self.gradInput
end