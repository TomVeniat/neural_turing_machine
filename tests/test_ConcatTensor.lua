require 'nn'
require 'nngraph'
require '../modules/ConcatTensor'


print('\nTest ConcatTensor :')
local t = torch.range(1,5)

local concat = nn.ConcatTensor(3)
print(concat:forward(t))

local grad = torch.rand(3,5)
print(concat:backward(t, grad))

print('Graphs :')

local t_size = 5
local n_concat = 3

local in1 = nn.Identity()()

local dimensions = {}

for i=1,n_concat do
	dimensions[i] = nn.Identity()(in1)
end

local join = nn.JoinTable(1)(dimensions)

local g1 = nn.gModule({in1},{join})


local in1 = nn.Identity()()
local res = nn.ConcatTensor(n_concat)(in1)

local g2 = nn.gModule({in1},{res})

local input = torch.rand(1,t_size)

print('Forward')
print(g1:forward(input))
print(g2:forward(input))

local grad = torch.rand(n_concat, t_size)

print('Backward')
print(g1:backward(input, grad))
print(g2:backward(input, grad))