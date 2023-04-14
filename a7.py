from neural import *

'''
    XOR test
'''

xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

xor = NeuralNet(2, 1, 1)
xor.train(xor_data)
print(xor.test_with_expected(xor_data))

print("\n\n\n\n\n\n")

'''
    Voter Opinion Test
'''

voter_opinion = [
    ([.9, .6, .8, .3, .1],[1]),
    ([.8, .8, .4, .6, .4],[1]),
    ([.7, .2, .4, .6, .3],[1]),
    ([.5, .5, .8, .4, .8],[0]),
    ([.3, .1, .6, .8, .8],[0]),
    ([.6, .3, .4, .3, .6],[0])
]
    
von = NeuralNet(5, 6, 1)

von.train(voter_opinion)
#print(von.test_with_expected(voter_opinion))

#Evaluate w/ test data
test_data = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]

print(f"case 1: {von.evaluate(test_data[0])}")
print(f"case 2: {von.evaluate(test_data[1])}")
print(f"case 3: {von.evaluate(test_data[2])}")
print(f"case 4: {von.evaluate(test_data[3])}")
print(f"case 5: {von.evaluate(test_data[4])}")





