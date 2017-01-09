require 'nn'
require 'nngraph'
require '../modules/PowScalar'


print('\nTest PowScalar :')
local t = torch.rand(5)
local s = torch.Tensor{2}

local pow = nn.PowScalar()

print(pow:forward({s,t}))

local grad = torch.rand(5)
print(unpack(pow:backward({s,t}, grad)))




-- print('Graphs :')
