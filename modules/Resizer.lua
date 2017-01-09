
--Simple module allowing to hardcode values into a network, used only for testing 
local Resizer, _ = torch.class('nn.Resizer', 'nn.Module')

function Resizer:__init(size)
	self.size = size
end

function Resizer:updateOutput(input)

	self.output = input:view(unpack(self.size))

	return self.output
end


function Resizer:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:view(input:size())

	return self.gradInput
end