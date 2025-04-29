# LLM Learning

This project is dedicated to exploring and learning about Large Language Models (LLMs). It includes experiments, examples, and resources to better understand how LLMs work and how to use them effectively.

## Purpose

The purpose of this project is to:
- Gain hands-on experience with LLMs.
- Experiment with various use cases and applications.
- Document findings and best practices for working with LLMs.

## How to Use

1. **Clone the Repository**  
    ```bash
    git clone https://github.com/your-username/llm-learning.git
    cd llm-learning
    ```

2. **Configure VirtualEnv and Install Dependencies**  
    Ensure you have Python installed, then install the required packages:
    ```bash
    python3 -m venv -env
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Modify data and run the script**  
    Navigate to the `data` directory. Inside there are two files, one is for training the model (feedback.cvs) and the other one (examples.csv) is used as an input for applying it after trained.
    Modify them as will and then execute from the root of the project:
    ```bash
    python -m src.sentiment_analysys_sklearn --data-file=data/feedback.csv --examples-file=data/examples.csv
    ```

4. **Contribute**  
    Feel free to contribute by adding new experiments or improving documentation.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`