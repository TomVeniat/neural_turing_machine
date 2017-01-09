local PowScalar, _ = torch.class('nn.PowScalar', 'nn.Module')

function PowScalar:updateOutput(input)
	local scal_tensor = input[1]
	local vec = input[2]

	local scal
	if scal_tensor:dim() == 1 then
		scal = scal_tensor[1]
	else
		scal = scal_tensor[1][1]
	end
	self.output = torch.pow(vec,scal)

	return self.output
end


function PowScalar:updateGradInput(input, gradOutput)
	local scal_tensor = input[1]
	local vec = input[2]

	local scal
	if scal_tensor:dim() == 1 then
		scal = scal_tensor[1]
	else
		scal = scal_tensor[1][1]
	end

	local size
	if vec:dim() == 1 then
		size = vec:size(1)
	else
		size = vec:size(2)
	end

	self.gradInput = {}

	-- No gradient for negative values of input vector, to be confirmed.
	local grad_pow = 0
	for i=1,size do
		if vec[i] > 0 then
			grad_pow = grad_pow + torch.log(vec[i]) * torch.pow(vec[i], scal) * gradOutput[i]
		end
	end

	self.gradInput[1] = torch.Tensor({grad_pow}):view(scal_tensor:size())
	self.gradInput[2] = torch.cmul(gradOutput, scal * torch.pow(vec,scal - 1))

	return self.gradInput
end