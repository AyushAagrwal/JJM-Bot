document.addEventListener("DOMContentLoaded", function () {
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const chatBox = document.getElementById("chat-box");
  const languageSelect = document.getElementById("language-select");
  const youtubeSearchBtn = document.getElementById("youtube-search-btn");

  let selectedLanguage = "en"; // Default language is English

  function validateInput(input) {
    input = input.trim();

    if (!input) {
      alert("Input cannot be empty.");
      return false;
    }

    const prohibitedPattern = /<[^>]*>/g;
    if (prohibitedPattern.test(input)) {
      alert("Input contains prohibited characters.");
      return false;
    }

    const maxLength = 500;
    if (input.length > maxLength) {
      alert(`Input cannot be longer than ${maxLength} characters.`);
      return false;
    }

    return true;
  }

  languageSelect.addEventListener("change", function () {
    selectedLanguage = languageSelect.value;
    clearChat();
    appendMessage(
      "assistant",
      `Language switched to ${
        languageSelect.options[languageSelect.selectedIndex].text
      }`
    );
  });

  sendBtn.addEventListener("click", sendMessage);

  userInput.addEventListener("keydown", function (event) {
    if (event.keyCode === 13) {
      // 13 is the key code for Enter
      sendMessage();
    }
  });

  clearBtn.addEventListener("click", function () {
    clearChat();
  });

  youtubeSearchBtn.addEventListener("click", function () {
    const question = userInput.value.trim();
    if (validateInput(question)) {
      appendMessage("user", question);
      fetchYoutubeLinks(question);
    }
    userInput.value = "";
  });

  function sendMessage() {
    const question = userInput.value.trim();
    if (validateInput(question)) {
      disableInput();
      appendMessage("user", question);
      fetchResponse(question);
    }
    userInput.value = "";
  }

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
    enableInput();

    if (!result.available && result.faq_available) {
      appendSuggestedQuestions(result.suggestions);
    }
    // Store the context for feedback
    document.getElementById("user-input").dataset.question = question;
    document.getElementById("user-input").dataset.answer = result.answer;
  }

  async function fetchYoutubeLinks(question) {
    appendTypingIndicator();
    try {
      const response = await fetch("/youtube_search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, language: selectedLanguage }),
      });
      const result = await response.json();
      removeTypingIndicator();
      appendYoutubeLinks(result.youtube_videos);
    } catch (error) {
      console.error("Error fetching YouTube links:", error);
      removeTypingIndicator();
    }
  }

  function appendYoutubeLinks(videos) {
    if (videos.length === 0) {
      appendMessage("assistant", "No YouTube videos found.");
    } else {
      const headerDiv = document.createElement("div");
      headerDiv.className = "youtube-header";
      headerDiv.textContent = "Suggested YouTube Links:";
      chatBox.appendChild(headerDiv);

      videos.forEach((video) => {
        const videoDiv = document.createElement("div");
        videoDiv.className = "video-link";

        const videoTitle = document.createElement("a");
        videoTitle.href = video.url;
        videoTitle.textContent = filterEmojis(video.title); // Filter emojis from title
        videoTitle.target = "_blank";

        videoDiv.appendChild(videoTitle);
        chatBox.appendChild(videoDiv);
      });
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  }

  // Function to filter emojis from text
  function filterEmojis(text) {
    return text.replace(
      /[\u{1F600}-\u{1F6FF}|\u{1F300}-\u{1F5FF}|\u{1F900}-\u{1F9FF}|\u{2600}-\u{26FF}|\u{2700}-\u{27BF}]/gu,
      ""
    );
  }

  function appendSuggestedQuestions(suggestions) {
    const suggestionDiv = document.createElement("div");
    suggestionDiv.className = "suggestions";

    const suggestionHeader = document.createElement("div");
    suggestionHeader.className = "suggestion-header";
    suggestionHeader.textContent = "Suggested Questions:";
    suggestionDiv.appendChild(suggestionHeader);

    const suggestionSelect = document.createElement("select");
    suggestionSelect.className = "suggestion-select";
    suggestionSelect.size = 5;

    suggestions.forEach((suggestion) => {
      const option = document.createElement("option");
      option.value = suggestion;
      option.textContent = suggestion;
      suggestionSelect.appendChild(option);

      option.addEventListener("click", async function () {
        disableInput();
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
        enableInput();
      });
    });

    suggestionDiv.appendChild(suggestionSelect);
    chatBox.appendChild(suggestionDiv);
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
    typingDot3.className = "typing";

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

  function disableInput() {
    userInput.disabled = true;
    sendBtn.disabled = true;
    clearBtn.disabled = true;
    youtubeSearchBtn.disabled = true;
  }

  function enableInput() {
    userInput.disabled = false;
    sendBtn.disabled = false;
    clearBtn.disabled = false;
    youtubeSearchBtn.disabled = false;
    userInput.focus();
  }
});

document.getElementById("feedback-btn").addEventListener("click", function () {
  document.getElementById("feedback-form").classList.toggle("hidden");
});

document
  .getElementById("submit-feedback-btn")
  .addEventListener("click", function () {
    const feedback = document.getElementById("feedback-text").value.trim();

    if (feedback === "") {
      alert("Feedback cannot be empty.");
      return;
    }

    const userQuestion = document.getElementById("user-input").value;
    const answer = document.querySelector(
      "#chat-box .message.assistant span:last-child"
    ).innerText;

    fetch("/submit_feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_question: userQuestion,
        answer: answer,
        feedback: feedback,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          alert("Feedback submitted successfully!");
          document.getElementById("feedback-form").classList.add("hidden");
          document.getElementById("feedback-text").value = "";
        } else {
          alert("Failed to submit feedback. Please try again.");
        }
      });
  });
