from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# Load tokenizer & model TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,   # Dùng torch.float32 nếu gặp vấn đề với float16
    device_map="auto"
)

# Khởi tạo lịch sử hội thoại, giới hạn tối đa số câu hỏi
def init_history(max_history=5):
    return [{
        "role": "system",
        "content": (
            "Bạn là một trợ lý ảo thông minh tên là Thắng. "
            "Bạn nói tiếng Việt một cách lịch sự, thân thiện và hữu ích. "
            "Khi người dùng nói 'Xin chào' hoặc hỏi 'Bạn là ai?', bạn nên trả lời: 'Tôi là Thắng, trợ lý ảo của bạn.' "
            "Luôn luôn cố gắng hỗ trợ người dùng tốt nhất."
        )
    }]

# Hàm giới hạn số lượng lịch sử trò chuyện
def limit_history(history, max_length=5):
    return history[-max_length:]

# Hàm tạo phản hồi từ mô hình
def generate_reply(user_input, history):
    if not history or not isinstance(history, list):
        history = init_history()

    # Giới hạn lịch sử để giảm số token đầu vào
    history = limit_history(history, max_length=5)

    # Thêm câu hỏi của người dùng vào lịch sử
    history.append({"role": "user", "content": user_input})

    # Tạo prompt dạng chat
    prompt = "".join([f"{entry['content']}\n" for entry in history])

    # Chuyển đổi prompt thành input_ids
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Sinh phản hồi từ mô hình
    output = model.generate(
        input_ids,
        max_new_tokens=150,  # Giới hạn số token để tăng tốc độ
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Trích xuất phản hồi và loại bỏ phần "assistant:"
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # Loại bỏ phần "assistant:" nếu có
    response = response.replace("assistant:", "").strip()

    history.append({"role": "assistant", "content": response})
    
    return history, history

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Trợ lý ảo Thắng – Phiên bản TinyLlama nhanh nhẹ!")

    chatbot = gr.Chatbot(type="messages", label="Thắng AI")
    state = gr.State(init_history())  # Khởi tạo lịch sử hội thoại

    with gr.Row():
        msg = gr.Textbox(placeholder="Nhập câu hỏi tại đây...", show_label=False)
        send_btn = gr.Button("Gửi")

    send_btn.click(generate_reply, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(generate_reply, inputs=[msg, state], outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()
