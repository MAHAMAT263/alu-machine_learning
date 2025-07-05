#!/usr/bin/env python3
from __future__ import annotations
import os

semantic_search = __import__('3-semantic_search').semantic_search
question_answer_from_doc = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers user questions from a set of reference documents using semantic search and BERT QA.

    Args:
        corpus_path (str): Path to a folder containing text or markdown reference files.
    """
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        # Step 1: Use semantic search to find the most relevant reference document
        reference = semantic_search(corpus_path, question)

        # Step 2: Try to extract an answer from the reference document
        answer = question_answer_from_doc(question, reference)

        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
