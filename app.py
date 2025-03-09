import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime
import time

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class ChatBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        if self.model is None:
            log_message("Loading base model and tokenizer...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
            
            log_message("Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                "phi2-finetuned/checkpoint-8928",
                device_map="auto"
            )
            log_message("Model loaded successfully!")
    
    def generate_response(self, message, history):
        self.load_model()
        
        # Format input like training data
        prompt = f"Human: {message}\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                top_k=50,
                min_length=20,
            )
        
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        response = response.split("-I-")[0].strip()
        return response

# Initialize chatbot
chatbot = ChatBot()

# Update the custom CSS with more space optimizations
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.message {
    padding: 0.3rem 1rem !important;  /* Further reduced padding */
    border-radius: 5px !important;
    margin-bottom: 0.3rem !important;  /* Further reduced margin */
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    font-size: 0.85rem !important;  /* Even smaller font */
}
.user-message {
    background: linear-gradient(135deg, #6e8efb, #4a6cf7) !important;
    color: white !important;
}
.assistant-message {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4) !important;
}
/* Tighter markdown spacing */
.md-container {
    margin: 0.3rem 0 !important;
}
/* Optimize input area */
.input-textbox textarea {
    font-size: 0.85rem !important;
    padding: 0.3rem !important;
    min-height: 2.2rem !important;
    line-height: 1.2 !important;
}
/* Reduce container gaps */
.chatbot-container {
    gap: 0.3rem !important;
}
/* Optimize header spacing */
h1 {
    margin: 0.5rem 0 !important;
    font-size: 1.5rem !important;
}
/* Compact button styling */
button {
    min-height: 2.2rem !important;
    font-size: 0.85rem !important;
}
/* Remove extra padding from containers */
.container {
    padding: 0.3rem !important;
}
.gradio-container {
    padding: 1rem !important;
}
/* Optimize row spacing */
.row {
    gap: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}
"""

# Update the interface section with more compact layout
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(scale=1, min_width=0):  # Added min_width=0 for tighter layout
        gr.Markdown(
            """# 🤖 Phi-2 Fine-tuned Chatbot
            Welcome to the interactive chat interface!""")
        
        # More compact two-column feature list
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=0):
                gr.Markdown("- 💡 Intelligent responses\n- 🎨 Beautiful UI")
            with gr.Column(scale=1, min_width=0):
                gr.Markdown("- ⚡ Real-time generation\n- 🔄 Memory")
    
    # Optimize chat components spacing
    chatbot_component = gr.Chatbot(
        value=[],
        height=350,
        show_label=False,
        type="messages",
        container=True,
        elem_classes="chatbot-container"
    )
    
    # More compact input area
    with gr.Row(equal_height=True):
        msg = gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=7,
            show_label=False,
            elem_classes="input-textbox",
            min_width=0
        )
        submit = gr.Button("Send 📤", variant="primary", scale=1, min_width=0)
        clear = gr.Button("Clear 🗑️", scale=1, min_width=0)

    # Update custom CSS first
    custom_css += """
    /* Style for example buttons */
    .example-grid {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 0.5rem !important;
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }
    .example-button {
        font-size: 0.75rem !important;
        padding: 0.4rem 0.6rem !important;
        text-align: left !important;
        white-space: normal !important;
        height: auto !important;    
        background: linear-gradient(135deg, #f6f8fa, #e9ecef) !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 5px !important;
    }
    .example-button:hover {
        background: linear-gradient(135deg, #e9ecef, #dee2e6) !important;
        transform: translateY(-1px) !important;
    }
    .gradio-container {
        max-width: 1200px !important;  /* Increased width */
    }
    """

    # Example questions in two rows with three prompts each
    gr.Markdown("### Try these examples:")
    
    # First row of examples
    with gr.Row(elem_classes="example-grid"):
        gr.Button(
            "What is artificial intelligence?",
            elem_classes="example-button"
        ).click(lambda: "What is artificial intelligence?", outputs=msg)
        gr.Button(
            "Explain quantum computing in simple terms",
            elem_classes="example-button"
        ).click(lambda: "Explain quantum computing in simple terms", outputs=msg)
        gr.Button(
            "Write a short poem about nature",
            elem_classes="example-button"
        ).click(lambda: "Write a short poem about nature", outputs=msg)
    
    # Second row of examples
    with gr.Row(elem_classes="example-grid"):
        gr.Button(
            "What are the benefits of exercise?",
            elem_classes="example-button"
        ).click(lambda: "What are the benefits of exercise?", outputs=msg)
        gr.Button(
            "How does machine learning differ from traditional programming?",
            elem_classes="example-button"
        ).click(lambda: "How does machine learning differ from traditional programming?", outputs=msg)
        gr.Button(
            "What are the ethical considerations in AI development?",
            elem_classes="example-button"
        ).click(lambda: "What are the ethical considerations in AI development?", outputs=msg)

    def user_message(user_message, history):
        # Validate empty input
        if not user_message or user_message.strip() == "":
            gr.Warning("Please enter a message!")
            return user_message, history
        return "", history + [{"role": "user", "content": user_message}]

    def bot_message(history):
        if len(history) > 0:
            response = chatbot.generate_response(history[-1]["content"], history)
            history.append({"role": "assistant", "content": response})
            return history

    def clear_chat():
        return []

    # Set up event handlers with validation
    msg.submit(
        user_message,
        [msg, chatbot_component],
        [msg, chatbot_component],
        api_name=False  # Prevent empty API calls
    ).then(
        bot_message,
        chatbot_component,
        chatbot_component,
        api_name=False
    )

    submit.click(
        user_message,
        [msg, chatbot_component],
        [msg, chatbot_component],
        api_name=False  # Prevent empty API calls
    ).then(
        bot_message,
        chatbot_component,
        chatbot_component,
        api_name=False
    )

    clear.click(clear_chat, None, chatbot_component)

# Launch the app
if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    ) 