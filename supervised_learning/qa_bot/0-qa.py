#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(question, reference):
    """
    Finds an answer to a question within a reference document using BERT QA.

    Args:
        question (str): The question to be answered.
        reference (str): The reference document containing the answer.

    Returns:
        str or None: The extracted answer, or None if not found.
    """
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize question and reference
    inputs = tokenizer.encode_plus(question, reference, add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]

    # Run model
    outputs = model([inputs["input_ids"],
                     inputs["token_type_ids"],
                     inputs["attention_mask"]])

    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Get most probable start and end of answer
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)

    if start_index >= len(input_ids) or end_index >= len(input_ids) or start_index > end_index:
        return None

    tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index + 1])
    answer = tokenizer.convert_tokens_to_string(tokens)

    return answer if answer.strip() else None
