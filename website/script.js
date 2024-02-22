function askQuestion() {
    var input = document.getElementById("user-query");
    var chatBox = document.getElementById("chat-box");
    var userText = "<div>User: " + input.value + "</div>";

    var botReply = "<div>Bot: " + getResponse(input.value) + "</div>";
    chatBox.innerHTML += userText + botReply;
    input.value = ""; // Clear input after sending
    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
}

function getResponse(query) {
    query = query.toLowerCase().trim();
    switch (query) {
        case "who was the first president of the united states?":
            return "George Washington was the first president of the United States.";
        case "when did the second world war end?":
            return "The Second World War ended in 1945.";
        default:
            return "Sorry, I don't have the answer to that question. Please ask another.";
    }
}
