let userData = {
  message: null,
  file: {}
};
document.addEventListener("DOMContentLoaded", function() {
  const promptForm = document.getElementById("prompt-form");
  const promptInput = document.getElementById("prompt-input");
  const sendButton = document.getElementById("send-prompt-btn");
  promptForm.addEventListener("submit", function(e) {
    e.preventDefault(); 
    const message = promptInput.value.trim();
    if (!message) return;
    promptInput.value = "";
    const userMessageDiv = createMessageElement(message, "user");
    document.querySelector(".chats-container").appendChild(userMessageDiv);

   
    const botMessageDiv = createMessageElement("", "bot", true);
    document.querySelector(".chats-container").appendChild(botMessageDiv);
    userData.message = message;
    generateResponse(botMessageDiv);

    scrollToBottom();
  });
  promptInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      promptForm.dispatchEvent(new Event("submit"));
    }
  });
});
function createMessageElement(message, sender, isLoading = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}-message`;
  const messageContent = document.createElement("div");
  messageContent.className = "message-content";
  const avatar = document.createElement("div");
  avatar.className = `message-avatar ${sender}-avatar`;
  avatar.innerHTML = sender === "user" ? "👤" : "🤖";

  const messageText = document.createElement("div");
  messageText.className = "message-text";
  if (isLoading) {
    messageText.textContent = "Thinking...";
    messageDiv.classList.add("loading");
    document.body.classList.add("bot-responding");
  } else {
    messageText.textContent = message;
  }

  messageContent.appendChild(avatar);
  messageContent.appendChild(messageText);
  messageDiv.appendChild(messageContent);

  return messageDiv;
}
function typingEffect(text, textElement, messageDiv) {
  textElement.textContent = "";
  const words = text.split(" ");
  let wordIndex = 0;

  function typeNextWord() {
    if (wordIndex < words.length) {
      textElement.textContent += (wordIndex === 0 ? "" : " ") + words[wordIndex];
      wordIndex++;
      setTimeout(typeNextWord, 100);
    } else {
      messageDiv.classList.remove("loading");
      document.body.classList.remove("bot-responding");
    }
  }

  typeNextWord();
}
function scrollToBottom() {
  const chatsContainer = document.querySelector(".chats-container");
  chatsContainer.scrollTop = chatsContainer.scrollHeight;
}

const generateResponse = async (botMsgDiv) => {
  const textElement = botMsgDiv.querySelector(".message-text");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userData.message || "",
        file: userData.file && userData.file.data ? userData.file : null,
      }),
    });

    const data = await response.json();
    if (response.ok) {
      const responseText = data.response.replace(/\*\*([^*]+)\*\*/g, "$1").trim();
      typingEffect(responseText, textElement, botMsgDiv);
    } else {
      throw new Error(data.error || "Something went wrong.");
    }
  } catch (error) {
    textElement.textContent = error.message;
    textElement.style.color = "#d62939";
    botMsgDiv.classList.remove("loading");
    document.body.classList.remove("bot-responding");
    scrollToBottom();
  } finally {
    userData.file = {};
  }
};
