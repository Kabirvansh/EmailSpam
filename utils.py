import re
import os
import pandas as pd

def preprocessor(e):
    """Preprocess text by converting to lowercase and removing non-alphabetic characters."""
    result = []
    for k in e:
        if k.isalpha():  
            result.append(k.lower()) 
        else:
            result.append(' ')  
    return ''.join(result).strip()

def read_spam():
    """Read spam emails from the dataset."""
    category = 'spam'
    directory = 'enron1/spam'
    return read_category(category, directory)

def read_ham():
    """Read legitimate (ham) emails from the dataset."""
    category = 'ham'
    directory = 'enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    """Read emails from a specific category and directory."""
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({
                    'name': filename, 
                    'content': content, 
                    'category': category
                })
            except:
                print(f'skipped {filename}')
    return emails
