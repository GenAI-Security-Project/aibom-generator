// OWASP AIBOM Generator - Common Scripts

/* === reCAPTCHA Handling (Index Page) === */
function onSubmit(token) {
    if (!token) {
        console.error("reCAPTCHA token is missing.");
        return;
    }

    console.log("reCAPTCHA callback executed with token.");

    // Set the token in the hidden input
    var tokenInput = document.getElementById('g-recaptcha-response');
    if (tokenInput) {
        tokenInput.value = token;
        console.log("Token set in input.");
    } else {
        console.error("g-recaptcha-response input not found.");
        return;
    }

    // Disable button and change text
    var button = document.getElementById('generate-button');
    if (button) {
        button.disabled = true;
        button.textContent = 'Generating...';
    }

    // Submit the form
    console.log("Submitting form");
    var form = document.getElementById('sbom-form');
    if (form) {
        form.submit();
    } else {
        console.error("Form sbom-form not found.");
    }
}

// Add Enter key support for form submission (Index Page)
document.addEventListener('DOMContentLoaded', function () {
    var modelInput = document.querySelector('input[name="model_id"]');
    if (modelInput) {
        modelInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                var btn = document.getElementById('generate-button');
                if (btn) btn.click();
            }
        });
    }
});


/* === Result Page Functions === */

function switchTab(tabId) {
    // Hide all tab contents
    var tabContents = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    // Deactivate all tabs
    var tabs = document.getElementsByClassName('aibom-tab');
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove('active');
    }

    // Activate the selected tab and content
    var content = document.getElementById(tabId);
    if (content) content.classList.add('active');

    var selectedTab = document.querySelector('.aibom-tab[onclick="switchTab(\'' + tabId + '\')"]');
    if (selectedTab) selectedTab.classList.add('active');
}

function toggleCollapsible(element) {
    element.classList.toggle('active');
    var content = element.nextElementSibling;
    if (content) {
        content.classList.toggle('active');

        if (content.classList.contains('active')) {
            content.style.maxHeight = content.scrollHeight + 'px';
        } else {
            content.style.maxHeight = '0';
        }
    }
}

/**
 * Downloads a JSON object as a file.
 * @param {Object|string} content - The JSON object or string to download.
 * @param {string} filename - The name of the file to save as.
 */
function downloadJSON(content, filename) {
    var jsonString = (typeof content === 'string') ? content : JSON.stringify(content, null, 2);
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(jsonString);

    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", filename || "aibom.json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

// Initialize collapsible sections (Result Page)
document.addEventListener('DOMContentLoaded', function () {
    var collapsibles = document.getElementsByClassName('collapsible');
    for (var i = 0; i < collapsibles.length; i++) {
        // Remove existing onclick to avoid double firing if inline remains, 
        // but cleaner to attach here if not attached inline.
        // However, HTML onclick="toggleCollapsible(this)" is common pattern.
        // If the HTML has onclick, we don't need addEventListener here unless we remove onclick from HTML.
        // For now, let's assumes HTML calls toggleCollapsible(this).
        // Initialization of state might be needed though.
    }
    // If elements start collapsed, no JS init needed other than event handlers.
});

// Validate Hugging Face URL or Model ID (Index Page)
document.addEventListener('DOMContentLoaded', function () {
    var modelInput = document.getElementById('model-input');
    var generateButton = document.getElementById('generate-button');

    if (modelInput && generateButton) {
        function validateInput() {
            var value = modelInput.value.trim();
            // Check if it's a valid HF URL (starts with https://huggingface.co/)
            // OR a valid org/repo identifier (e.g. openai/whisper-tiny)
            var isUrl = value.startsWith('https://huggingface.co/');
            // Basic regex for org/repo: alphanumeric, dots, dashes, underscores
            var isModelId = /^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$/.test(value);

            if (isUrl || isModelId) {
                generateButton.disabled = false;
                generateButton.style.cursor = 'pointer';
                generateButton.style.opacity = '1';
            } else {
                generateButton.disabled = true;
                generateButton.style.cursor = 'not-allowed';
                generateButton.style.opacity = '0.6';
            }
        }

        modelInput.addEventListener('input', validateInput);
        // Initial check
        validateInput();
    }
});
