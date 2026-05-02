import gradio as gr
from inference import run  # Import your grader-compliant function

def chat_function(message, history):
    # Gradio 5.x passes history as list[dict] already — inference.py handles it
    try:
        response = run(message, history)
        # Gradio renders HTML so <tool_call> tags get hidden by the browser.
        # Wrap in a code block for display only — the grader calls run() directly.
        if "<tool_call>" in response:
            return f"```\n{response}\n```"
        return response
    except Exception as e:
        return f"[error] {e}"

# Create the UI
demo = gr.ChatInterface(
    fn=chat_function,
    title="📱 Pocket-Agent: On-Device Tool Caller",
    description=(
        "Testing the quantized Qwen3-0.6B GGUF (~420 MB, Q4_K_M). "
        "Try asking for weather, currency conversion, or SQL queries!"
    ),
    examples=[
        "What's the weather in Karachi?",
        "Convert 500 USD to PKR",
        "Show me all users from the employees table",
        "Add a meeting to my calendar for tomorrow named 'Project Sync'"
    ]
)

if __name__ == "__main__":
    # share=True creates a public URL so judges can access it from their browser
    demo.launch(share=True)