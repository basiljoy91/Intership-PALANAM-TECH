chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classify") {
        fetch('http://localhost:5002/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: request.text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => sendResponse(data))
        .catch(error => {
            console.error('Fetch error:', error);
            sendResponse({ 
                error: error.message,
                category: 'Error',
                confidence: 0
            });
        });
        
        return true; // Required for async response
    }
});