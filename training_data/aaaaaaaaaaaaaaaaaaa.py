import math, random, time, pygame

learning_speed = 0.2

#Takes any number and translates it to a number between 0 and 1
def sigmoid(x):
  return 1 / (1 + math.e**(x * -1))

def sigmoid_prime(x):
  return x * (1 - x)

def soft_max(x):
  return math.e**x

def soft_max_prime(x):
  return math.e * x


def real_answer(x):
  # return (math.sin(x / 2) / 2) + 0.5
  # return 1 / (1 + math.e**(x * -1))
  return x**5 + 4 * x / 19
  # return 3 * x**4

#A class neuron which has an index indicating which neuron within a layer it is, a number 
# of input neurons which determines the number of weights and biases we need
class neuron:
  def __init__(self, index, num_of_inputs, value = 0):
    self.index = index
    self.num_of_weights = num_of_inputs
    self.weights = []
    self.biases = []
    self.weights_adjust = []
    self.biases_adjust = []
    self.effect_on_error = []
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
  def forward(self, use_relu = True):
    for current_neuron in range(len(self.neurons)):
      total = 0
      for neuron in range(len(self.input.neurons)):
        value = self.input.neurons[neuron].value
        corr_weight = self.neurons[current_neuron].weights[neuron]
        corr_bias = self.neurons[current_neuron].biases[neuron]
        total = total + (value * corr_weight) + corr_bias
      if use_relu == True:
        self.neurons[current_neuron].value = sigmoid(total)
      else:
        self.neurons[current_neuron].value = soft_max(total)

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
    hidden_layers = self.hidden_layers
    for layer in hidden_layers:
      if hidden_layers[-1] == hidden_layers[hidden_layers.index(layer)]:
        layer.forward(False)
      else:
        layer.forward()

  def calculate_cost(self, correct_out):
    difference = correct_out - network.hidden_layers[-1].neurons[0].value
    return difference**2

  def backward_pass(self, target_output):
    hidden_layers = self.hidden_layers
    for layer in hidden_layers[::-1]:
      for neuron in layer.neurons:
        neuron.weights_adjust.append([])
        neuron.biases_adjust.append([])
        for weight in neuron.weights:
          if hidden_layers.index(layer) != 0:
            previous_layer = hidden_layers[hidden_layers.index(layer) - 1]
          else:
            previous_layer = self.input_layer

          if hidden_layers[-1] == hidden_layers[hidden_layers.index(layer)]:
            weight_index = neuron.weights.index(weight)
            prev_corr_neuron = previous_layer.neurons[weight_index]


            par_der = prev_corr_neuron.value * soft_max_prime(neuron.value) * self.calculate_cost(target_output) * 2
            neuron.weights_adjust[-1].append(par_der)

            par_der_bi = soft_max_prime(neuron.value) * self.calculate_cost(target_output) * 2
            neuron.biases_adjust[-1].append(par_der_bi)

            par_der_va = weight * soft_max_prime(neuron.value) * self.calculate_cost(target_output) * 2
            print(weight, self.calculate_cost(target_output), "par_der_va")
            prev_corr_neuron.effect_on_error.append(par_der_va)

          else:
            average_effect = 0
            for effect in neuron.effect_on_error:
              average_effect += effect
            average_effect = average_effect / len(neuron.effect_on_error)

            # print(average_effect * 2, prev_corr_neuron.value)

            weight_index = neuron.weights.index(weight)
            prev_corr_neuron = previous_layer.neurons[weight_index]

            par_der = prev_corr_neuron.value * sigmoid_prime(neuron.value) * average_effect
            neuron.weights_adjust[-1].append(par_der)

            par_der_bi = sigmoid_prime(neuron.value) * average_effect
            neuron.biases_adjust[-1].append(par_der_bi)

            par_der_va = weight * average_effect * sigmoid_prime(neuron.value)
            prev_corr_neuron.effect_on_error.append(par_der_va)





  def adjust_weights(self):
    hidden_layers = self.hidden_layers
    for layer in hidden_layers[::-1]:
      for neuron in layer.neurons:
        for weight in neuron.weights:
          weight_index = neuron.weights.index(weight)
          
          weight_adjustment = 0
          for adjust in neuron.weights_adjust:
            weight_adjustment += adjust[weight_index]
          weight_adjustment = weight_adjustment / len(neuron.weights_adjust)
          neuron.weights[weight_index] += weight_adjustment * -1 * learning_speed
          
          bias_adjustment = 0
          for adjust in neuron.biases_adjust:
            bias_adjustment += adjust[weight_index]
          bias_adjustment = bias_adjustment / len(neuron.biases_adjust)
          neuron.biases[weight_index] += bias_adjustment * -1 * learning_speed

        neuron.weights_adjust = []
        neuron.biases_adjust = []

            
          

pygame.init()

window = pygame.display.set_mode((1000, 600))

#Yes
neurons_per_layer = [1, 3, 3, 3, 1]
network = neural_network(neurons_per_layer)

for i in range(100):
  window.fill((0, 0, 0))
  for x in range(1000):
    pygame.draw.rect(window, ((255, 255, 255)), (x, real_answer(x / 1000) * 600, 2, 2))
    pygame.display.flip()

  for batch in range(500):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    batch_list = []
    for i in range(30):
      batch_list.append(random.random())

    answer_list = []
    for x in batch_list:
      answer_list.append(real_answer(x))

    my_answers = []
    for input, correct in zip(batch_list, answer_list):
      network.forward(input)
      my_answers.append(network.hidden_layers[-1].neurons[0].value)
      network.backward_pass(correct)


    network.adjust_weights()

    # cost = 0
    # for input_, my_ans in zip(batch_list, my_answers):
    #   cost += network.calculate_cost(real_answer(input_), my_ans)
    # print(cost / len(batch_list))

    for input_, my_ans in zip(batch_list, my_answers):
      pygame.draw.rect(window, ((100, 100, 255)), (input_ * 1000, my_ans * 600, 2, 2))
      pygame.display.flip()
    
