document.getElementById("analyzeBtn").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: scrapeAndSendText
    });
  });
});

function scrapeAndSendText() {
  const text = document.body.innerText;
  fetch("http://localhost:5001/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  })
    .then(response => response.json())
    .then(data => {
      alert(`Sentiment Score: ${data.score}`);
    });
}
