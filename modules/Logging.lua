local Logging, _ = torch.class('nn.Logging', 'nn.Module')


function Logging:__init(name, active)
	self.name = name
	if active == nil then
		active = true
	end
	self.active = active
end


function Logging:updateOutput(input)
	self.output = input
	if self.active then 
		print(self.name .. ' :')
		print(input)
	end
	return self.output
end


function Logging:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
	if self.active then 
		print(self.name .. ' back :')
		print(gradOutput)
	end
	return self.gradInput
end