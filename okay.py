import PyPDF2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import absl.flags as flags


from bert import tokenization as bert_tokenization

# tf.disable_v2_behavior()
# FLAGS = tf.flags.FLAGS

FLAGS = flags.FLAGS
FLAGS(['okay.py'])
tf.compat.v1.disable_eager_execution()
# Load the pre-trained BERT model from TensorFlow Hub
bert_module = hub.load("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")

# Load the PDF file
pdf_file = open('sample.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)
pdf_text = ""

# Extract the text from the PDF file
# for page_num in range(pdf_reader.numPages):
#     page = pdf_reader.getPage(page_num)
#     pdf_text += page.extractText()
for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    pdf_text += page.extract_text()

pdf_file.close()
# Tokenize the text using BERT's tokenizer
tokenizer = bert_tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)
tokens = tokenizer.tokenize(pdf_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Pad the token ids to a fixed length
max_seq_length = 128
token_ids = token_ids[:max_seq_length]
input_mask = [1] * len(token_ids)
while len(token_ids) < max_seq_length:
    token_ids.append(0)
    input_mask.append(0)

# Run the BERT model on the input
input_word_ids = np.array([token_ids])
input_mask = np.array([input_mask])
segment_ids = np.zeros_like(input_mask)
pooled_output, sequence_output = bert_module([input_word_ids, input_mask, segment_ids])

# Define a function to answer questions based on the BERT output
def answer_question(question):
    # Tokenize the question using BERT's tokenizer
    question_tokens = tokenizer.tokenize(question)
    question_token_ids = tokenizer.convert_tokens_to_ids(question_tokens)

    # Pad the token ids to a fixed length
    question_token_ids = question_token_ids[:max_seq_length - 2]
    question_token_ids = [101] + question_token_ids + [102]
    input_mask = [1] * len(question_token_ids)
    while len(question_token_ids) < max_seq_length:
        question_token_ids.append(0)
        input_mask.append(0)

    # Run the BERT model on the input
    input_word_ids = np.array([question_token_ids])
    input_mask = np.array([input_mask])
    segment_ids = np.zeros_like(input_mask)
    _, question_output = bert_module([input_word_ids, input_mask, segment_ids])

    # Compute the similarity between the question and the PDF text
    similarity_scores = np.dot(sequence_output[0], question_output[0]) / (
                np.linalg.norm(sequence_output[0]) * np.linalg.norm(question_output[0]))
    most_similar_sentence_index = np.argmax(similarity_scores)

    # Extract the most similar sentence and return it as the answer
    most_similar_sentence_tokens = tokens[most_similar_sentence_index:min(most_similar_sentence_index + 10, len(tokens))]
    answer = tokenizer.convert_tokens_to_string(most_similar_sentence_tokens)
    return answer

# Ask a question and get the answer
question = "What is the capital of France?"
answer = answer_question(question)
print(answer)
