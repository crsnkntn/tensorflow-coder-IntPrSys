import subprocess
import json

def main():
    spec = "inputs = { 'in1': [[1, 6], [3, 8]], 'in2': [[2, 4], [5, 7]], 'in3': [[3, 8], [4, 7]], 'in4': [[6, 5], [2, 1]] } # The single desired output tensor. output = [[-5, -3], [-2, 0]] # A list of relevant scalar constants (if any). constants = [] # An English description of the tensor manipulation. description = 'Subtract the max and min of tensors compared to the transpose of other tensors'"
    response = subprocess.run(['sh', './test.sh', spec], capture_output=True, text=True)

    print(response.stdout)
    data = json.loads(response.stdout)

    # Extract the 'content' string from the nested structure
    content_string = data['choices'][0]['message']['content']
    print("content_string: ", content_string)

    # The 'content' itself is a JSON string, so parse it again
    functions = json.loads(content_string)

    # Print or use the 'functions' list
    print(functions)


if __name__ == "__main__":
    main()