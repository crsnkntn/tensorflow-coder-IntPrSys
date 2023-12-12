#!/bin/bash
nld="$@"

# Define the prompt as a variable
prompt="The following is a description of a TensorFlow program: \"$nld\". The following is a briefly expanded description of the described program: "

jsonPayload=$(jq -n \
          --arg content "$prompt" \
          '{model: "gpt-3.5-turbo-1106", messages: [{role: "user", content: $content}], temperature: 0.7}')

# Your API Key should be stored securely, not in the script
api_key="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"

# Making the API request
response=$(curl -s https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $api_key" \
  -d "$jsonPayload")

# Output handling
echo $response
