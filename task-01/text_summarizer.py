from transformers import pipeline
import textwrap

# Load summarization pipeline from Hugging Face
summarizer = pipeline("summarization")

# Function to summarize input text
def summarize_text(input_text, max_length=50, min_length=25):
    """
    Summarize the input text using a transformer-based model.

    Args:
        input_text (str): The text to be summarized.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: A concise summary of the input text.
    """
    if not input_text.strip():
        return "Input text is empty. Please provide valid text."

    try:
        summary = summarizer(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {e}"

# Example usage
if __name__ == "__main__":
    print("Welcome to the Text Summarizer!\n")
    print("Enter your text below (type 'exit' to quit):\n")

    while True:
        user_input = input("Input Text: ")
        if user_input.lower() == 'exit':
            print("Exiting the Text Summarizer. Goodbye!")
            break

        print("\nSummarized Text:")
        print(textwrap.fill(summarize_text(user_input), width=80))
        print("\n" + "-"*80 + "\n")
