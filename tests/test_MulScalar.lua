require 'nn'
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
print(unpack(mScal:backward({vec1,scal},grad)))



local vec1 = torch.rand(10) 
local grad = torch.rand(10) 
local truc = nn.MulConstant(scal[1])

print('Forward :')
print(mScal:forward({vec1,scal}))
print(truc:forward(vec1)) 

print('Backward :')
print(unpack(mScal:backward({vec1,scal},grad)))
print(truc:backward(vec1,grad)) 