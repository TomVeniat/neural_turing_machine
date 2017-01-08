local MulScalar, _ = torch.class('nn.MulScalar', 'nn.Module')

function MulScalar:updateOutput(input)
	local scal_tensor = input[1]
	local vec = input[2]

	local scal
	if scal_tensor:dim() == 1 then
		scal = scal_tensor[1]
	else
		scal = scal_tensor[1][1]
	end
	self.output = vec * scal

	return self.output
end


function MulScalar:updateGradInput(input, gradOutput)
	local scal_tensor = input[1]
	local vec = input[2]

	local scal
	if scal_tensor:dim() == 1 then
		scal = scal_tensor[1]
	else
		scal = scal_tensor[1][1]
	end

	self.gradInput = {}
	self.gradInput[1] = torch.Tensor({vec:dot(gradOutput)}):resize(scal_tensor:size())
	self.gradInput[2] = gradOutput * scal

	return self.gradInput
end