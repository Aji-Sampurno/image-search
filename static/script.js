const uploadArea = document.getElementById('upload-area');
const uploadBox = document.querySelector('.upload-box');
const fileInput = document.getElementById('file-input');
const cameraInput = document.getElementById('camera-input');
const browseBtn = document.getElementById('browse-btn');
const cameraBtn = document.getElementById('camera-btn');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const loader = document.getElementById('loader');
const resultsGrid = document.getElementById('results-grid');

// Event Listeners
browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

cameraBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    cameraInput.click();
});

uploadBox.addEventListener('click', () => fileInput.click());
clearBtn.addEventListener('click', clearSelection);

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

cameraInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload an image file.");
        return;
    }

    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.classList.add('hidden');
        previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Perform Search
    performSearch(file);
}

function clearSelection() {
    fileInput.value = '';
    previewImage.src = '';
    uploadBox.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultsGrid.innerHTML = '';
}

async function performSearch(file) {
    // UI State
    loader.classList.remove('hidden');
    resultsGrid.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        const data = await response.json();
        renderResults(data.results);

    } catch (error) {
        console.error(error);
        resultsGrid.innerHTML = '<p style="color:red; text-align:center;">An error occurred while searching.</p>';
    } finally {
        loader.classList.add('hidden');
    }
}

function renderResults(results) {
    if (!results || results.length === 0) {
        resultsGrid.innerHTML = '<p style="color:var(--text-secondary); text-align:center;">No matching results found.</p>';
        return;
    }

    results.forEach(item => {
        // Backend now returns 0-100
        const scorePercent = item.score.toFixed(1);

        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="card-img-wrapper">
                <!-- In a real app, item.url would be valid. 
                     For this demo, if it's local, we might not be able to load it unless served.
                     We'll put a placeholder if image load fails. -->
                <img src="${item.url}" alt="${item.filename}" onerror="this.src='https://placehold.co/400x400/222/FFF?text=Image'">
            </div>
            <div class="card-info">
                <div class="card-title" title="${item.filename}">${item.filename}</div>
                <div class="similarity-bar-bg">
                    <div class="similarity-bar-fill" style="width: ${Math.min(scorePercent, 100)}%"></div>
                </div>
                <div class="card-meta">
                    <span>Similarity</span>
                    <span class="score">${scorePercent}%</span>
                </div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}
