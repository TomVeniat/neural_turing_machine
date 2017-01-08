
--Simple module allowing to hardcode values into a network, used only for testing 
local Hijack, _ = torch.class('nn.Hijack', 'nn.Module')

function Hijack:__init(value)
	self.value = value
end

function Hijack:updateOutput(input)


	self.output = self.value or input

	return self.output
end


function Hijack:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput

	return self.gradInput
end