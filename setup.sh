# author: steeve Laquitaine

# setup virtual environment
conda create -n intent python==3.6.14
conda activate intent
pip install -r src/intent/requirements.txt

# set nltk environment variables
export NLTK_DATA=$(pwd)/"data/06_models/nltk_data" 