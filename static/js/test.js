document.addEventListener("DOMContentLoaded", function () {
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const chatBox = document.getElementById("chat-box");
  const languageSelect = document.getElementById("language-select");
  let selectedLanguage = "en";
  let suggestionDiv = null;

  languageSelect.addEventListener("change", function () {
    selectedLanguage = this.value;
    clearChat();
    appendMessage(
      "assistant",
      `Language switched to ${
        languageSelect.options[languageSelect.selectedIndex].text
      }`
    );
  });

  function sendMessage() {
    const userInput = document.getElementById("user-input");
    const question = userInput.value.trim();
    if (question) {
      appendMessage("user", question);
      fetchResponse(question);
    }
    userInput.value = "";
  }

  sendBtn.addEventListener("click", sendMessage);

  document
    .getElementById("user-input")
    .addEventListener("keydown", function (event) {
      if (event.keyCode === 13) {
        sendMessage();
      }
    });

  clearBtn.addEventListener("click", function () {
    clearChat();
  });

  function clearChat() {
    chatBox.innerHTML = `
        <div class="message assistant">
          <img src="/static/images/bot.png" alt="Bot" class="avatar" />
          <div class="message-content"><span>Ask me a question based on the guidelines</span></div>
        </div>`;
  }

  function appendMessage(role, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement("img");
    avatar.className = "avatar";
    avatar.src =
      role === "user" ? "/static/images/user.png" : "/static/images/bot.png";
    avatar.alt = role;

    const textDiv = document.createElement("div");
    textDiv.className = "message-content";
    const textSpan = document.createElement("span");
    textSpan.innerHTML = formatText(content);
    textDiv.appendChild(textSpan);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(textDiv);

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function formatText(text) {
    return text
      .replace(/\*\*Challenges:\*\*/g, "<b>Challenges:</b>")
      .replace(/\*\*Solutions:\*\*/g, "<b>Solutions:</b>")
      .replace(/\*\*/g, "")
      .replace(/\n/g, "<br>");
  }

  async function fetchResponse(question) {
    appendTypingIndicator();
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language: selectedLanguage }),
    });
    const result = await response.json();
    removeTypingIndicator();
    appendMessage("assistant", result.answer);

    if (!result.available && result.faq_available) {
      appendSuggestedQuestions(result.suggestions);
    } else if (!result.available) {
      const googleButton = document.createElement("button");
      googleButton.textContent = "Search on Google";
      googleButton.className = "google-btn";
      googleButton.onclick = async function () {
        googleButton.textContent = "Searching in Google...";
        googleButton.disabled = true;
        await googleSearch(question);
        googleButton.remove();
      };
      chatBox.appendChild(googleButton);
    }
  }

  function appendSuggestedQuestions(suggestions) {
    if (suggestionDiv) {
      suggestionDiv.remove();
    }

    suggestionDiv = document.createElement("div");
    suggestionDiv.className = "suggestions";

    const suggestionHeader = document.createElement("div");
    suggestionHeader.className = "suggestion-header";
    suggestionHeader.textContent = "Suggested Questions:";
    suggestionDiv.appendChild(suggestionHeader);

    const suggestionSelect = document.createElement("select");
    suggestionSelect.className = "suggestion-select";
    suggestionDiv.appendChild(suggestionSelect);

    suggestions.forEach((suggestion) => {
      const option = document.createElement("option");
      option.value = suggestion;
      option.textContent = suggestion;
      suggestionSelect.appendChild(option);
    });

    suggestionSelect.addEventListener("change", async function () {
      const selectedQuestion = this.value;
      appendMessage("user", selectedQuestion);
      const response = await fetch("/faq_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: selectedQuestion,
          language: selectedLanguage,
        }),
      });
      const result = await response.json();
      appendMessage("assistant", result.answer);
      suggestionDiv.remove();
    });

    chatBox.appendChild(suggestionDiv);
  }

  async function googleSearch(query) {
    appendMessage("assistant", "Searching in Google...");
    const response = await fetch("/google_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, language: selectedLanguage }),
    });
    const result = await response.json();
    removeTypingIndicator();
    appendMessage("assistant", result.answers.join("<br>"));
  }

  function appendTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "message assistant typing-indicator";

    const avatar = document.createElement("img");
    avatar.className = "avatar";
    avatar.src = "/static/images/bot.png";
    avatar.alt = "Bot";

    const typingDot1 = document.createElement("div");
    typingDot1.className = "typing";

    const typingDot2 = document.createElement("div");
    typingDot2.className = "typing";

    const typingDot3 = document.createElement("div");
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(typingDot1);
    typingDiv.appendChild(typingDot2);
    typingDiv.appendChild(typingDot3);

    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function removeTypingIndicator() {
    const typingIndicator = document.querySelector(".typing-indicator");
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }
});
