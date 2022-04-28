import math, random, time, pygame

learning_speed = 0.5

#Takes any number and translates it to a number between 0 and 1
def sigmoid(x):
    return 1 / (1 + math.e**(x * -1))

def sigmoid_prime(x):
  return x*(1 - x)

def real_answer(x):
  return x**5 + 4 * x / 19
  # return (math.sin(x * 5) / 2) + 0.5

#A class neuron which has an index indicating which neuron within a layer it is, a number 
# of input neurons which determines the number of weights and biases we need
class neuron:
  def __init__(self, index, num_of_inputs, value = 0):
    self.index = index
    self.num_of_weights = num_of_inputs
    self.weights = []
    self.biases = []
    self.value = value
    for weight in range(num_of_inputs):
      self.weights.append(random.random())
    for bias in range(num_of_inputs):
      self.biases.append(random.random())
    if len(self.weights) == 0:
      self.weights.append(1)

#The input layer class is just like a layer class but it doesn't have an input of its own and 
# so no weights and biases assosiated with it
class input_layer:
  def __init__(self, num_of_neurons):
    self.neurons = []
    self.num_of_neurons = num_of_neurons
    self.step = 0
    for ne_num in range(num_of_neurons):
      self.neurons.append(neuron(ne_num, 0))

#Sets the value of the only neuron in the input layer to the random number in the first input list
  def forward(self, input_):
    for current_neuron in self.neurons:
      current_neuron.value = input_

#A layer object has its own index to determine where in the hidden layers it is, the number of 
# neurons and the layer which is going to be the input 
class layer:
  def __init__(self, index, num_of_neurons, input):
    self.index = index
    self.neurons = []
    self.num_of_neurons = num_of_neurons
    self.input = input
    for ne_num in range(num_of_neurons):
      self.neurons.append(neuron(ne_num, input.num_of_neurons))

#For every neuron in the layer, it takes the input neurons, multiplies them by their weight and adds 
# their bias and passes that through a sigmoid function to map it to a number between 0 and 1
  def forward(self):
    for current_neuron in range(len(self.neurons)):
      total = 0
      for neuron in range(len(self.input.neurons)):
        value = self.input.neurons[neuron].value
        corr_weight = self.neurons[current_neuron].weights[neuron]
        corr_bias = self.neurons[current_neuron].biases[neuron]
        total = total + (value * corr_weight) + corr_bias
      self.neurons[current_neuron].value = sigmoid(total)

#The neural network is a class so I can step everything forward all at once and have less inputs
#list of neurons is actually a list of the number of neurons per layer but idk how to shorten that
class neural_network:
  def __init__(self, list_of_neurons):
    self.input_layer = input_layer(list_of_neurons[0])
    self.hidden_layers = []
    for i in range(len(list_of_neurons) - 1):
      if i == 0:
        prev_layer = self.input_layer
      else:
        prev_layer = self.hidden_layers[-1]
      self.hidden_layers.append(layer(i+1, list_of_neurons[i+1], prev_layer))

  #Really just steps forward the input layer, then all the hidden layers
  def forward(self, input_):
    self.input_layer.forward(input_)
    for layer in self.hidden_layers:
      layer.forward()

  def calculate_cost(self, correct_out, actual_out):
    difference = correct_out - actual_out
    return difference**2

  def backward_pass(self, target_output):
    hidden_layers = self.hidden_layers
    for layer in hidden_layers[::-1]:
      for neuron in layer.neurons:
        for weight in neuron.weights:
          if hidden_layers.index(layer) != 0:
            previous_layer = hidden_layers[hidden_layers.index(layer) - 1]
          else:
            previous_layer = self.input_layer

          weight_index = neuron.weights.index(weight)
          prev_corr_neuron = previous_layer.neurons[weight_index]

          par_der = prev_corr_neuron.value * sigmoid_prime(neuron.value) * ((neuron.value - target_output) * 2)
          neuron.weights[neuron.weights.index(weight)] += par_der * -1 * learning_speed

          par_der_bi = sigmoid_prime(neuron.value) * ((neuron.value - target_output) * 2)
          neuron.biases[weight_index] += par_der_bi * -1 * learning_speed

pygame.init()

window = pygame.display.set_mode((1000, 800))

for x in range(1000):
  pygame.draw.rect(window, ((255, 255, 255)), (x, real_answer(x / 1000) * 700, 2, 2))
  pygame.display.flip()
time.sleep(2)

#Yes
neurons_per_layer = [1, 10, 10, 10, 1]
network = neural_network(neurons_per_layer)

for i in range(1000000):
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      quit()

  random_x = random.random()
  network.forward(random_x)
  network.backward_pass(real_answer(random_x))

  # cost = network.calculate_cost(real_answer(random_x), network.hidden_layers[-1].neurons[0].value)
  answer = network.hidden_layers[-1].neurons[0].value

  # print(real_answer(random_x), network.hidden_layers[-1].neurons[0].value, cost, i)
  if i > 3000:
    pygame.draw.rect(window, ((100, 100, 255)), (random_x * 1000, answer * 700, 2, 2))
    pygame.display.flip()

  # if cost < 1e-15:
  #   quit()


