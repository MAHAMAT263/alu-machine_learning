#!/usr/bin/env python3

def main():
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        user_input = input("Q: ").strip()
        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break
        else:
            print("A:")

if __name__ == "__main__":
    main()
