from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,  # Use bitsandbytes for memory efficiency
    trust_remote_code=True
)

# Create text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Formal sentence input
formal_sentence = """
“Today, I'm going to improve my speaking experience with chatting and talking with you. Are you ready for that? For this purpose?”
"""

# Create prompt
prompt = f"""
You are an expert in rewriting formal English into casual, natural spoken English.
Return only one informal sentence. Do not add any explanations or extra text.
Your output must not contain any grammar mistakes.

Formal: {formal_sentence}
Informal:"""


# Generate
output = text_generator(
    prompt,
    max_new_tokens=60,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

# Extract and print
generated_text = output[0]['generated_text']
informal_version = generated_text.split("Informal:")[-1].strip()
print('**************************************************************************************************')
print("Informal version:", informal_version)
