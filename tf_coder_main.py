# Copyright 2021 The TF-Coder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A script for using TF-Coder (an alternative to using the Colab notebook).

Usage:
1. Edit `get_problem()` to specify your problem.
2. If desired, edit `get_settings()` to specify settings for TF-Coder.
3. Run this file, e.g., `python3 tf_coder_main.py`.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is faster than GPU.

from absl import app  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module

from tf_coder.benchmarks import all_benchmarks

def get_problem():
  """Specifies a problem to run TF-Coder on. Edit this function!"""
  # A dict mapping input variable names to input tensors.
  inputs = {
      'in1': [[1, 2], [3, 4]],
      'in2': [[5, 6], [7, 8]]
  }

  # The single desired output tensor.
  output = [[1, 5, 2, 6, 3, 7, 4, 8]]

  # A list of relevant scalar constants (if any).
  constants = []

  # An English description of the tensor manipulation.
  description = 'merge two tensors then reshape them using tensor 3'

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
  colab_interface.run_value_search_from_colab(
      inputs, output, constants, description, settings)


def print_supported_operations():
  """Run this function to print all supported operations."""
  colab_interface.print_supported_operations()


def main(unused_argv):
  # It takes several seconds to load the models.
  colab_interface.warm_up()

  inputs, output, constants, description = get_problem()
  settings = get_settings()

  benchmarks = all_benchmarks.all_benchmarks()

  for bm in benchmarks:
    print("Searching for: ", bm.target_program)
    run_tf_coder(bm.examples[0].inputs, bm.examples[0].output, bm.constants, bm.description, settings)
    


if __name__ == '__main__':
  app.run(main)
