import gzip, numpy as np, math, random, time, pygame

def get_images(path):
  with gzip.open(path, 'r') as f:
    # first 4 bytes is a magic number
    magic_number = int.from_bytes(f.read(4), 'big')
    # second 4 bytes is the number of images
    image_count = int.from_bytes(f.read(4), 'big')
    # third 4 bytes is the row count
    row_count = int.from_bytes(f.read(4), 'big')
    # fourth 4 bytes is the column count
    column_count = int.from_bytes(f.read(4), 'big')
    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    image_data = f.read()
    images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
    return images


def get_labels(path):
  with gzip.open(path, 'r') as f:
    # first 4 bytes is a magic number
    magic_number = int.from_bytes(f.read(4), 'big')
    # second 4 bytes is the number of labels
    label_count = int.from_bytes(f.read(4), 'big')
    # rest is the label data, each label is stored as unsigned byte
    # label values are 0 to 9
    label_data = f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    return labels

def mean(list_x):
    total = 0
    for i in list_x:
        total = total + i

    return total / len(list_x)

def max(list_):
  max_value = None
  for num in list_:
    if (max_value is None or num > max_value):
      max_value = num

  return max_value

def sigmoid(x):
  return 1 / (1 + math.e**x)

def sigmoid_prime(x):
  return sigmoid(x) / (1 - sigmoid(x))

def ReLU(x):
  if x < 0:
    return 0
  else:
    return x

def ReLU_prime(x):
  if x < 0:
    return 0
  else:
    return 1

def softmax(x, output_list):
  total_den = 0
  for i in output_list:
    total_den += (math.e**i)

  return (math.e**x) / total_den

def softmax_prime(x, output_list):
  return softmax(x, output_list) * (1 - (softmax(x, output_list)))



class neuron():
  def __init__(self, index, num_of_inputs, activation = 0):
    self.index = index
    self.activation = activation
    self.num_of_inputs = num_of_inputs
    self.total_value = 0
    self.weights = []
    self.weights_adjust = []
    self.biases_adjust = []
    self.value_derivatives = []
    self.bias = random.uniform(0, 1)

    for i in range(self.num_of_inputs):
      self.weights.append(random.uniform(0, 1))
      self.weights_adjust.append([])

  def forward(self, activation_function, inputs):
    self.total_value = 0
    for act, wgt in zip(inputs, self.weights):
      self.total_value += (act * wgt)

    self.total_value += self.bias

    if activation_function == "sigmoid":
      self.activation = sigmoid(self.total_value)
    elif activation_function == "ReLU":
      self.activation = ReLU(self.total_value)
    elif activation_function == "softmax":
      self.activation = self.total_value

  def special_forward(self, total_output):
    self.activation = softmax(self.total_value, total_output)


class layer():
  def __init__(self, index, num_of_inputs, num_of_neurons):
    self.index = index
    self.num_of_neurons = num_of_neurons
    self.num_of_inputs = num_of_inputs
    self.activation_function = "sigmoid"
    self.neurons = []

    for i in range(num_of_neurons):
      self.neurons.append(neuron(i, num_of_inputs))

  def forward(self, inputs):
    if self.activation_function == "softmax":
      self.output = []
      total_layer_values = []
      for neuron in self.neurons:
        neuron.forward(self.activation_function, inputs)
        total_layer_values.append(neuron.total_value)
      for neuron in self.neurons:
        neuron.special_forward(total_layer_values)
        self.output.append(neuron.activation)
    else:
      self.output = []
      for neuron in self.neurons:
        neuron.forward(self.activation_function, inputs)
        self.output.append(neuron.activation)


class neural_network():
  def __init__(self, num_of_inputs, neurons_per_layer):
    self.layers = []

    for i in range(len(neurons_per_layer)):
      if i == 0:
        inp = num_of_inputs
      else:
        inp = len(self.layers[-1].neurons)
      self.layers.append(layer(i, inp, neurons_per_layer[i]))

  def forward(self, inputs):
    self.inputs = inputs
    self.layers[0].forward(inputs)

    for i in range(len(self.layers) - 1):
      self.layers[i + 1].forward(self.layers[i].output)

  def backward(self, target):
    #starting from the last layer, moving to the first layer:
    for layer in self.layers[::-1]:
      for neuron in layer.neurons:
        if layer == self.layers[-1]:
          #if we're looking at the target neuron, the target is 1, otherwise its 0.1
          if neuron.index == target:
            if neuron.activation < 0.0001:
              neuron.activation = 0.0001
            derivative_error = -2 * math.log(neuron.activation)
          else:
            if neuron.activation > 0.9999:
              neuron.activation = 0.9999
            derivative_error = -2 * math.log(1 - neuron.activation)
        else:
          #if we aren't dealing with the last layer, we reuse calculations from the previous layer as the error
          derivative_error = mean(neuron.value_derivatives)
          neuron.value_derivatives = []

        for weight in range(len(neuron.weights)):
          if layer != self.layers[0]:
            previous_neuron = self.layers[layer.index - 1].neurons[weight]
            previous_neuron_value = previous_neuron.activation
          else:
            previous_neuron_value = self.inputs[weight]

          if layer.activation_function == "sigmoid":
            activation_derivative = sigmoid_prime(neuron.total_value)
          elif layer.activation_function == "ReLU":
            activation_derivative = ReLU_prime(neuron.total_value)
          elif layer.activation_function == "softmax":
            total_layer_values = []
            for neuron_2 in layer.neurons:
              total_layer_values.append(neuron_2.total_value)
            activation_derivative = softmax_prime(neuron.total_value, total_layer_values)
            # print(activation_derivative, derivative_error, previous_neuron_value)

          bias_derivative = derivative_error * activation_derivative
          weight_derivative = bias_derivative * previous_neuron_value

          neuron.weights_adjust[weight].append(weight_derivative)
          neuron.biases_adjust.append(bias_derivative)
          if layer != self.layers[0]:
            previous_neuron.value_derivatives.append(bias_derivative)

  def adjust_wb(self, learning_speed):
    for layer in self.layers:
      for neuron in layer.neurons:
        try:
          neuron.bias += (mean(neuron.biases_adjust) * learning_speed)
        except:
          print(layer.index)
          quit()
        for weight in range(len(neuron.weights)):
          neuron.weights[weight] += (mean(neuron.weights_adjust[weight]) * learning_speed)

    for layer in self.layers:
      for neuron in layer.neurons:
        neuron.weights_adjust = []
        neuron.biases_adjust = []
        for i in range(neuron.num_of_inputs):
          neuron.weights_adjust.append([])



learning_speed = -1
neurons_per_layer = [30, 10]
network = neural_network(784, neurons_per_layer)
network.layers[1].activation_function = "ReLU"
# network.layers[2].activation_function = "ReLU"
network.layers[-1].activation_function = "softmax"dddd

images = get_images("training_data/train-images-idx3-ubyte.gz")
labels = get_labels("training_data/train-labels-idx1-ubyte.gz")

count = -1
while True:
  for j in range(5):
    count += 1
    if count > 5:
      count = 1
    current_image = images[count]
    current_label = labels[count]

    image_pixels = []
    for row in range(len(current_image)):
      for pixel in range(len(current_image[row])):
        image_pixels.append((current_image[row][pixel]) / 255)

    network.forward(image_pixels)
    network.backward(current_label)
    pick = network.layers[-1].output.index(max(network.layers[-1].output))

    cost = 0
    output = []
    for k in range(len(network.layers[-1].output)):
      output.append(round(network.layers[-1].output[k], 3))
      value = network.layers[-1].output[k]
      if value > 0.9999:
        value = 0.9999
      elif value < 0.0001:
        value = 0.0001
      if k == current_label:
        cost += -1 * math.log(value)
      else:
        cost += -1 * math.log(1 - value)

    print(current_label, pick, count, output, "cost =", cost**2)
  network.adjust_wb(learning_speed)

