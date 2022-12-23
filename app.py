import openai
import nltk

API_KEY = "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"

openai.api_key = API_KEY


def summarize_text(text, model, max_tokens=2500):
    # Split the text into sentences using the sent_tokenize function
    sentences = nltk.sent_tokenize(text)
    summary = ""
    # Split the text into chunks of approximately 3000 tokens
    num_tokens = 0
    chunk = []
    for sentence in sentences:
        # Estimate the number of tokens in the sentence
        sentence_tokens = len(sentence) // 4
        # If the number of tokens in the chunk exceeds the maximum, summarize the chunk and reset the variables
        if num_tokens + sentence_tokens > max_tokens:
            prompt = f"\n{'. '.join(chunk)}\n\nSummarize:"
            completions = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=700,
                n=1,
                stop=None,
                temperature=0.7,
            )
            summary += completions.choices[0].text.strip() + "\n\n"
            print(f"Summarized {num_tokens} tokens...")
            chunk = []
            num_tokens = 0
        # Add the sentence to the chunk and update the number of tokens
        chunk.append(sentence)
        num_tokens += sentence_tokens
    # Summarize the final chunk if it exists
    if chunk:
        prompt = f"{'. '.join(chunk)}\n\nSummarize:"
        completions = openai.Completion.create(
            engine=model, prompt=prompt, max_tokens=700, n=1, stop=None, temperature=0.7
        )
        summary += completions.choices[0].text.strip() + "\n\n"
    return summary


with open("text.txt", "r") as f:
    text = f.read()

summary = summarize_text(text, model="text-davinci-003")

print(summary)
print()


def analyse_sentiment(text, model="curie"):
    """Analyse the sentiment of a text using the GPT-3 model."""
    prompt = f"{text}\n\n Sentiment:"
    completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0.7,
    )
    sentiment = completions.choices[0].text
    return sentiment


sentiment = analyse_sentiment(summary, model="curie")
print("Sentiment:", sentiment)


def extract_key_figures(text, model="text-davinci-003"):
    """Extract key figures from a text using the GPT-3 model."""
    prompt = f"{text}\n\n Extract key figures:"
    completions = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    key_figures = completions.choices[0].text
    return key_figures


key_figures = extract_key_figures(summary, model="text-davinci-003")
print("Key figures:", key_figures)

# with open("summary.txt", "r") as f:
#     summary = f.read()

# sentiment = analyse_sentiment(summary, model="curie")
# print("Sentiment:", sentiment)

# key_figures = extract_key_figures(summary, model="text-davinci-003")
# print("Key figures:", key_figures)

# import PyPDF2

# all_text = ""

# # Open the PDF file in read-binary mode
# with open('paper.pdf', 'rb') as file:
#     # Create a PDF object
#     pdf = PyPDF2.PdfReader(file)

#     # Get the number of pages in the PDF

#     # Iterate through each page
#     for page in pdf.pages:
#         # Extract the text from the page
#         text = page.extract_text()
#         all_text += text + "\n\n"

# summary = summarize_text(all_text, model="text-davinci-003")
# print("Summary:", summary)
