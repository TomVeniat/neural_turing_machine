require 'nn'
require 'nngraph'
require '../modules/MulScalar'


print('\nTest MulScalar :')

local mScal = nn.MulScalar()

local vec1 = torch.range(1,5) 
local scal = torch.ones(1)*2
print('Vector :')
print(vec1)
print('Scalar :')
print(scal)

print('Forward :')
print(mScal:forward({vec1,scal}))

local grad = torch.Tensor({1,-1,1,-1,0})
print('Backward :')
print(unpack(mScal:backward({scal, vec1},grad)))



local vec1 = torch.rand(10) 
local grad = torch.rand(10) 
local truc = nn.MulConstant(scal[1])

print('Forward :')
print(mScal:forward({scal, vec1}))
print(truc:forward(vec1)) 

print('Backward :')
print(unpack(mScal:backward({scal, vec1},grad)))
print(truc:backward(vec1,grad)) 


print('Graphs :')
local scalar = torch.rand(1,1)
local vector = torch.rand(1,3)
local grad = torch.rand(1,3)

local in1_scal = nn.Identity()()
local in1_vec = nn.Identity()()

local scalars = {}

for i=1,vector:size(2) do
	scalars[i] = nn.Identity()(in1_scal)
end	

local full_scals = nn.JoinTable(2)(scalars)

local res = nn.CMulTable()({full_scals,in1_vec})

-- g1 is a simulation of MulScalar using existing modules.
local g1 = nn.gModule({in1_scal, in1_vec},{res})


local in2_scal = nn.Identity()()
local in2_vec = nn.Identity()()

local res = nn.MulScalar()({in2_scal,in2_vec})

local g2 = nn.gModule({in2_scal,in2_vec}, {res})

print(g1:forward({scalar,vector}))
print(g2:forward({scalar,vector}))

print(unpack(g1:backward({scalar,vector}, grad)))
print(unpack(g2:backward({scalar,vector}, grad)))