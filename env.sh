# Huggingface Hub variables
export HF_NAME="Jongwon"
export HF_PWD="Rlaakfdl13$"
export HF_TOKEN="api_eZARVsAzZWQsqAbNmoACoGWGwTtQMBnFNo"
export HF_PROJECT="t5-tiny-it"

# Variables for training the tokenizer and creating the config
export VOCAB_SIZE="32000"
export N_INPUT_SENTENCES="1000000" # Num of sentences to train the tokenizer
export DATASET="gsarti/clean_mc4_it" # Name of the dataset in the Huggingface Hub
export DATASET_CONFIG="tiny" # Config of the dataset in the Huggingface Hub 
export DATASET_SPLIT="train" # Split to use for training tokenizer and model
export TEXT_FIELD="text" # Field containing the text to be used for training
export CONFIG_TYPE="google/t5-v1_1-base" # Config that our model will use
export MODEL_PATH="data/${HF_PROJECT}" # Path to the model, e.g. here inside the mount
