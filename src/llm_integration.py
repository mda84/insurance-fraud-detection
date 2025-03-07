from transformers import pipeline
import requests
import anthropic

def generate_explanation_anthropic(claim_data: dict, api_key: str) -> str:
    """
    Uses Anthropic's API to generate an explanation for why the vehicle claim might be fraudulent.
    
    You must have an Anthropic API key and follow their API documentation.
    """

    # Create an Anthropic client with the provided API key.
    client = anthropic.Anthropic(api_key=api_key)

    # Build the conversation messages list according to Anthropic's API requirements.
    messages = [
        {
            "role": "user",
            "content": (
                "Analyze the following vehicle claim details and provide a concise explanation for why "
                "this claim might be fraudulent. Consider red flags such as discrepancies between the vehicle's age "
                "and the policyholder's age, absence of a police report, lack of witnesses, and other potential anomalies.\n\n"
                f"Claim Details: {claim_data}"
            )
        }
    ]

        
    # Use a supported model name (e.g., "claude-v1") with the messages API.
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        max_tokens=150,
        temperature=0.5,
    )
    
    return response.content[0].text

def generate_explanation(claim_data: dict) -> str:
    """
    Uses a pre-trained text generation model (LLM) to generate a detailed explanation
    for why the vehicle claim might be fraudulent, highlighting potential red flags.
    """
    generator = pipeline("text-generation", model="gpt2")
    prompt = (
        "You are an expert insurance fraud investigator. Analyze the following vehicle claim details carefully. "
        "Consider common fraud red flags such as inconsistencies between the vehicle's reported age and the policyholder's age, "
        "lack of a police report or witnesses, and any other anomalies in the claim. "
        "Provide a detailed and concise explanation for why this claim might be fraudulent.\n\n"
        f"Claim Details: {claim_data}\n\nExplanation:"
    )
    # Adjusting parameters: increasing max_length for a longer response and setting temperature to 0.2 for a balanced output.
    explanation = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.5)
    return explanation[0]['generated_text']

# Example usage:
if __name__ == "__main__":
    sample_claim = {
        'AgeOfVehicle': '3 years',
        'AgeOfPolicyHolder': '26 to 30',
        'NumberOfSuppliments': 'none',
        'NumberOfCars': '1 vehicle',
        'Year': '1994',
        'PoliceReportFiled': 'No',
        'WitnessPresent': 'No'
    }
    print("Generated Explanation GPT2:")
    print(generate_explanation(sample_claim))
    # Replace 'YOUR_ANTHROPIC_API_KEY' with your actual API key.
    explanation = generate_explanation_anthropic(sample_claim, api_key="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")
    print("Generated Explanation Anthropic:")
    print(explanation)