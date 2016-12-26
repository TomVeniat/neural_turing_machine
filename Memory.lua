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
	return e:forward (w)
end