#!/bin/sh

# Start elastic search in the background.
~/elasticsearch-8.7.1/bin/elasticsearch & # point to your installation of elasticsearch.

# Download the original autocast dataset.
wget https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz
tar -xf autocast.tar.gz

# Process the questions.
python preprocess_autocast_questions.py

# Download and preproccess the docs.
python preprocess_cc_news.py

# Get the documents.
python assign_reading.py --in_file train_questions.json --out_file train_reading.json --n_docs 16
python assign_reading.py --in_file test_questions.json --out_file test_reading.json --n_docs 16

# Clean up.
rm autocast.tar.gz
rm -r autocast
