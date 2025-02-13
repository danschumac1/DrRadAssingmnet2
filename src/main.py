import os
import openai
import matplotlib.pyplot as plt
import numpy as np
import json

from utils.api import load_env

CLIENT = openai.OpenAI(api_key=load_env())

def generate_response(prompt, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, max_tokens=200):
    response = CLIENT.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=True
    )
    return response

def main():
    prompt = "Write a rap about calculus 3, linear algebra, and probability and combinatorics"

    # Experiment with different parameters
    experiments = [
        {"temperature": 0.5, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 1.0, "top_p": 1.0, "frequency_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.8, "frequency_penalty": 0.5},
    ]

    results = []

    for exp in experiments:
        response = generate_response(prompt, **exp)
        text = response.choices[0].message.content
        logprobs = response.choices[0].logprobs
        results.append({"params": exp, "text": text, "logprobs": logprobs})

        print("\n===== Generated Rap =====")
        print(text)
        print("========================\n")

        # save the rap to a file
        with open(f"./data/rap_tmp_{exp['temperature']}_topp_{exp['top_p']}_freq_{exp['frequency_penalty']}.txt", "w") as f:
            f.write(text)

    # Ensure output directory exists
    os.makedirs("./figures", exist_ok=True)

    # Plot overlapping probability distributions
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'green', 'red']
    alphas = [1.0, .8 , .5]  # Full opacity for first, less for second, even less for third

    for i, res in enumerate(results):
        logprobs = res["logprobs"].content  # Extract content list
        logprob_values = [token_logprob.logprob for token_logprob in logprobs]  # Extract log probabilities
        
        # Sort by log probabilities
        sorted_indices = np.argsort(logprob_values)  # Sort in ascending order
        sorted_logprobs = np.array(logprob_values)[sorted_indices]  # Reorder logprobs
        
        # Plot the probability distribution
        plt.bar(range(len(sorted_logprobs)), np.exp(sorted_logprobs), 
                color=colors[i], alpha=alphas[i], label=f'Experiment {i+1}: {res["params"]}')

    plt.title("Token Probability Distributions Across Three Raps")
    plt.ylabel("Probability")
    plt.xlabel("Token Index (Sorted by Log Probability)")
    plt.legend()
    plt.savefig("./figures/comparison_probability_distribution.png")
    plt.show()

    # Summarizing the findings
    print("\n=== Conclusion ===")
    for res in results:
        print(f"Parameters: {json.dumps(res['params'])}")
        print(f"Generated Rap: {res['text'][:200]}...\n")

if __name__ == "__main__":
    main()
