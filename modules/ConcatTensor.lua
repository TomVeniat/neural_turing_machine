local ConcatTensor, _ = torch.class('nn.ConcatTensor', 'nn.Module')

function ConcatTensor:__init(size)
	self.size = size
end

function ConcatTensor:updateOutput(input)
	-- local vec
	-- if input:dim() == 1 then
	-- 	vec = input:view(1,vec:size(1))
	-- else
	-- 	vec = input:view(1,vec:size(2))
	-- end

	self.output = input:repeatTensor(self.size,1)

	return self.output
end


function ConcatTensor:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:sum(1):resize(input:size())

	return self.gradInput
end