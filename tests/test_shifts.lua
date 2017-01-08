require 'nn'
require '../modules/Shifter'

local shift_range = {-1,0,1}
local shifter = nn.Shifter(shift_range)

local weigths = torch.Tensor({0,0.1,0.8,0.1})
local shifts = torch.Tensor({0,0.1,0.9})



print('\nTest rotate_vector :')
vec = torch.range(1,5)
print('initial :')
print(vec)

print('\nRotation of 2 :')
print(rotate_vector(vec,2))

print('\nRotation of -1 :')
print(rotate_vector(vec,-2))

print('\nTest Forward :')
print(shifter:forward({weigths,shifts}))


print('\nTest Backward :')
local grad = torch.Tensor({0,100,10,-5})
print(unpack(shifter:backward({weigths,shifts}, grad)))


shifter = nn.Shifter({-2,-1,0,1,2})

local n = 5000
local dim = 100 

local start = sys.clock()
for i = 1, n do
  local a = torch.randn(dim)
  local b = torch.randn(5)
  shifter:forward{a, b}

  local out_grad = torch.randn(dim)
  shifter:backward({a, b}, out_grad)
end
str = '%d iteration on %d dimensional vectors done in %fs'
print(str:format(n,dim,sys.clock() - start))