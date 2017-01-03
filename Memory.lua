local Memory = torch.class('Memory')

function Memory:__init(nbLocation, locationSize)
	self.nbLocation = nbLocation
	self.locationSize = locationSize
	self.mem = torch.range(1, locationSize * nbLocation):resize(nbLocation, locationSize)
end

function Memory:read(rWeightings)
	assert(rWeightings:size(1)==1 and rWeightings:size(2)==self.nbLocation,string.format('read Weightings must be 1x%d (1xN)', self.nbLocation))
	local r = rWeightings * self.mem
	return r
end

function Memory:getContentWeightings(key, beta)
	assert(key:size(1)==1 and key:size(2)==self.locationSize, 'Content adressing Key must be 1xM')
	assert(beta > 0 , string.format('Key strength must be greater than zero (here : %d)',beta))
	local expKey = key:repeatTensor(self.nbLocation,1)
	local c = nn.CosineDistance()
	local e = nn.SoftMax()
	local w = beta * c:forward({self.mem, expKey})
	return e:forward(w)
end

function Memory:shiftWeigthing(weights, st)
	-- assert(st:size(1)==3, 'Only -1, 0 and 1 shifts are allowed for now')
	size = weights:size(2)
	print(st)
	shiftMat = torch.zeros(size,size)

	for i=0,size-1 do
		print(i)
		for s, val in pairs(st) do
			local index = math.fmod(i + s + size, size) + 1 
			print(index)
			shiftMat[{{index},{i+1}}] = val
		end
	end

	return weights * shiftMat

end

function Memory:blendWeightings(prevWeights, contentWeights, gate)
	local gatedWeigths = gate * contentWeights + (1 - gate) * prevWeights
	return gatedWeigths
end

function Memory:sharpenWeigths(weights, gamma)
	local w = torch.Tensor(weights:size()):copy(weights):pow(gamma)
	return w / w:sum()	
end