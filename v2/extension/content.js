// Extract visible text from webpage
function extractVisibleText() {
    const elementsToRemove = document.querySelectorAll('script, style, noscript, iframe, svg');
    elementsToRemove.forEach(el => el.remove());

    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    let text = '';
    let node;
    while (node = walker.nextNode()) {
        const parentElement = node.parentElement;
        if (parentElement && isVisible(parentElement)) {
            text += ' ' + node.textContent.trim();
        }
    }

    return text.replace(/\s+/g, ' ').trim();
}

function isVisible(element) {
    const style = window.getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false;
    }
    if (element.offsetWidth === 0 || element.offsetHeight === 0) {
        return false;
    }
    return true;
}

async function classifyPage() {
    try {
        const text = extractVisibleText();
        if (!text) return;

        chrome.runtime.sendMessage(
            { action: "classify", text: text.substring(0, 10000) },
            (response) => {
                if (response.error) {
                    console.error('Classification error:', response.error);
                    showFallbackNotification('Error during classification');
                } else {
                    showClassificationResult(response);
                }
            }
        );
    } catch (error) {
        console.error('Error:', error);
        showFallbackNotification('Error processing page');
    }
}

function showClassificationResult(result) {
    const confidencePercent = Math.round(result.confidence * 100);
    const message = `This page is classified as ${result.category} (${confidencePercent}%)`;
    
    // Try to show as alert first
    try {
        alert(message);
    } catch (e) {
        // Fallback to custom notification if alert is blocked
        showCustomNotification(message, result.category);
    }
}

function showCustomNotification(message, category) {
    const colors = {
        'Safe': '#4CAF50',
        'Sensitive': '#FFC107',
        'Malicious': '#F44336',
        'Misleading': '#9E9E9E',
        'Error': '#2196F3',
        'Pending': '#9C27B0'
    };

    const notification = document.createElement('div');
    notification.style.position = 'fixed';
    notification.style.bottom = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px';
    notification.style.backgroundColor = colors[category] || '#2196F3';
    notification.style.color = 'white';
    notification.style.borderRadius = '5px';
    notification.style.zIndex = '999999';
    notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
    notification.innerHTML = `<strong>Content Safety:</strong><br>${message}`;
    
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5002);
}

function showFallbackNotification(message) {
    showCustomNotification(message, 'Error');
}

// Run classification when page loads
window.addEventListener('load', () => {
    setTimeout(classifyPage, 1500);
});