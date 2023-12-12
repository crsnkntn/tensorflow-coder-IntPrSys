import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is faster than GPU.

from absl import app  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.benchmarks import all_benchmarks

def get_problem2():
  """Specifies a problem to run TF-Coder on. Edit this function!"""
  # A dict mapping input variable names to input tensors.
  inputs = {
      'in1': [[1, 2], [3, 4]],
      'in2': [[5, 6], [7, 8]],
  }

  # The single desired output tensor.
  output = [[1, 2, 3, 4, 5, 6, 7, 8]]

  # A list of relevant scalar constants (if any).
  constants = []

  # An English description of the tensor manipulation.
  description = 'Merge two tensors and then reshape them using tensor 3'

  return inputs, output, constants, description

def get_problem3():
  """Specifies a problem to run TF-Coder on. Edit this function!"""
  # A dict mapping input variable names to input tensors.
  inputs = {
      'in1': [[1, 6], [3, 8]],
      'in2': [[2, 4], [5, 7]],
      'in3': [[3, 8], [4, 7]],
      'in4': [[6, 5], [2, 1]]
  }

  # The single desired output tensor.
  output = [[-5, -3], [-2, 0]]

  # A list of relevant scalar constants (if any).
  constants = []

  # An English description of the tensor manipulation.
  description = 'Subtract the maximum and minimum of tensors compared to the transpose of other tensors'

  return inputs, output, constants, description

def get_settings():
  """Specifies settings for TF-Coder. Edit this function!"""
  # How long to search for a solution, in seconds.
  time_limit = 300

  # How many solutions to find before stopping. If more than 1, the entire
  # search will slow down.
  number_of_solutions = 1

  # Whether solutions must use all inputs, at least one input, or no such
  # requirement. Choose one of "all inputs", "one input", "no restriction".
  solution_requirement = 'all inputs'
  assert solution_requirement in ['all inputs', 'one input', 'no restriction']

  return settings_module.from_dict({
      'timeout': time_limit,
      'only_minimal_solutions': False,
      'max_solutions': number_of_solutions,
      'require_all_inputs_used': solution_requirement == 'all inputs',
      'require_one_input_used': solution_requirement == 'one input',
  })


def run_tf_coder(inputs, output, constants, description, settings):
  """Runs TF-Coder on a problem, using the given settings."""
  # Results will be printed to standard output.
  result = colab_interface.run_value_search_from_colab(
      inputs, output, constants, description, settings)

  return result[1]


def main(unused_argv):
  # Load the models.
  # colab_interface.warm_up()

  inputs, output, constants, description = get_problem3()
  settings = get_settings()

  run_tf_coder(inputs, output, constants, description, settings)

if __name__ == '__main__':
  app.run(main)
