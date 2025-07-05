#!/usr/bin/env python3
from __future__ import annotations
import re

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers user questions based on a reference text using BERT QA.

    Args:
        reference (str): The reference document for answering questions
    """
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
