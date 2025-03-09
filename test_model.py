import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_model():
    log_message("Loading base model and tokenizer...")
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
    log_message("Loading fine-tuned adapter...")
    # Load the PEFT adapter
    model = PeftModel.from_pretrained(
        base_model,
        "phi2-finetuned/checkpoint-8928",  # Adjust path based on your saved checkpoint
        device_map="auto"
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, user_input, max_length=512):
    # Format input like training data
    prompt = f"Human: {user_input}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    log_message("Generating response...")
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,     # Prevent repetition
            no_repeat_ngram_size=3,     # Prevent phrase repetition
            eos_token_id=tokenizer.eos_token_id,
            top_k=50,                   # Added to limit vocabulary choices
            min_length=20,              # Added to ensure meaningful responses
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response
    # Split by "Assistant:" and take the last part
    response = response.split("Assistant:")[-1].strip()
    
    # Remove any "-I-" separators and duplicated content
    response = response.split("-I-")[0].strip()
    
    # Remove any repeated questions from the response
    if user_input in response:
        response = response.replace(user_input, "").strip()
    
    return response

def main():
    log_message("Initializing test program...")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model()
        
        log_message("Model loaded successfully! You can now chat with the model.")
        log_message("Type 'quit' or 'exit' to end the conversation.")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                log_message("Ending conversation...")
                break
            
            # Generate and print response
            response = generate_response(model, tokenizer, user_input)
            print("\nAssistant:", response)
            
    except Exception as e:
        log_message(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 