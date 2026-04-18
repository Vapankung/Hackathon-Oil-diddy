document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message");
  const sendBtn = document.getElementById("send-btn");

  function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function addMessage(role, text, label = null) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;

    if (label) {
      const labelDiv = document.createElement("div");
      labelDiv.className = "msg-label";
      labelDiv.textContent = label;
      div.appendChild(labelDiv);
    }

    const textNode = document.createElement("div");
    textNode.innerHTML = marked.parse(text);
    div.appendChild(textNode);

    chatBox.appendChild(div);
    scrollToBottom();
  }

  function createTypingBubble() {
    const loading = document.createElement("div");
    loading.className = "msg bot";
    loading.innerHTML = `
      <div class="msg-label">OilBot</div>
      <div class="typing">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;
    chatBox.appendChild(loading);
    scrollToBottom();
    return loading;
  }

  async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage("user", message, "You");
    messageInput.value = "";
    sendBtn.disabled = true;
    sendBtn.style.opacity = "0.75";

    const loading = createTypingBubble();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      const data = await res.json();
      loading.remove();
      addMessage("bot", data.answer || "No response.", "OilBot");
    } catch (error) {
      loading.remove();
      addMessage("bot", "Error connecting to backend.", "OilBot");
      console.error(error);
    } finally {
      sendBtn.disabled = false;
      sendBtn.style.opacity = "1";
      messageInput.focus();
      scrollToBottom();
    }
  }

  sendBtn.addEventListener("click", sendMessage);

  messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

  scrollToBottom();
});