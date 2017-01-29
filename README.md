# neural_turing_machine
Torch implementation of Neural turing Machine, as presented by Alex Graves, Greg Wayne &amp; Ivo Danihelka.

Original article : https://arxiv.org/abs/1410.5401

# Test

To test the different pre trained model just run the associated file :
 * Copy task : `load_assoc.lua`
 * Repeat copy task : `load_assoc.lua`
 * Association recall task : `load_assoc.lua`
 
 

#TO DO :
* Enhance Shifter module performances (caching shiftMat allows to reduce execution time by half).
* Implement a the GRU controller.
* Make training parameters configuration easier.
