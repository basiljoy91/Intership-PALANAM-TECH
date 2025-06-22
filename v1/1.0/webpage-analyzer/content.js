// Extract visible text
function extractVisibleText() {
  return Array.from(document.body.querySelectorAll("*"))
    .filter(el => el.childNodes.length === 1 && el.childNodes[0].nodeType === 3)
    .map(el => el.innerText.trim())
    .filter(text => text.length > 30)
    .join(" ");
}

// Send to backend API
async function analyzePage() {
  const text = extractVisibleText();

  const response = await fetch("http://127.0.0.1:5000/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  const result = await response.json();
  alert(`🛡️ Page is: ${result.category.toUpperCase()} (confidence: ${result.confidence})`);
}

window.onload = () => {
  setTimeout(analyzePage, 2000);  // wait for page to fully load
};
