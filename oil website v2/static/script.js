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
    textNode.innerHTML = text || "";
    div.appendChild(textNode);

    chatBox.appendChild(div);
    scrollToBottom();
  }

  async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage("user", message, "You");
    messageInput.value = "";

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })
      });

      const data = await res.json();

      addMessage("bot", data.answer || "No response.", "OilBot");

      if (data.graph_url) {
        const img = document.createElement("img");
        img.src = data.graph_url;
        img.style.maxWidth = "100%";
        img.style.marginTop = "10px";

        chatBox.appendChild(img);
      }

    } catch (err) {
      console.error(err);
      addMessage("bot", "Error connecting to backend.", "OilBot");
    }
  }

  sendBtn.addEventListener("click", sendMessage);

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });
});