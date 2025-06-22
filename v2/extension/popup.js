document.addEventListener('DOMContentLoaded', () => {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        const status = document.getElementById('status');
        
        // In a real implementation, you might store and display
        // the classification results here
        status.textContent = "Visit any page to see classification";
    });
});