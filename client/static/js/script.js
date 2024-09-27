const video = document.getElementById('camera-feed');
const canvas = document.getElementById('captured-image');
const captureBtn = document.getElementById('capture-btn');
const form = document.getElementById('upload-form');
const scanEffect = document.getElementById('scan-effect');
const statusDiv = document.getElementById('status');

// Request camera access with specific constraints
async function startCamera() {
    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user"
        }
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera:", err);
        statusDiv.textContent = "Camera access denied. Please check permissions.";
    }
}

startCamera();

// Capture image
captureBtn.addEventListener('click', () => {
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    captureBtn.disabled = true;

    // Animate scan effect
    gsap.to(scanEffect, {
        duration: 1,
        top: "100%",
        opacity: 1,
        ease: "power1.inOut",
        onComplete: processImage
    });
});

function processImage() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob and send to server
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'captured_image.jpg');

        fetch('/get_image', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
            .then(data => {
                console.log('Image saved:', data.filename);
                // You can update the UI here if needed
            }).catch(error => {
                console.error('Error saving image:', error);
                statusDiv.innerHTML = '<i class="fas fa-exclamation-circle text-red-500"></i> Error saving image';
            });
    }, 'image/jpeg');

    // Simulate face recognition process
    setTimeout(() => {
        statusDiv.innerHTML = '<i class="fas fa-check-circle text-green-500"></i> Welcome, User!';
        captureBtn.disabled = false;
    }, 2000);

    // Reset scan effect
    gsap.set(scanEffect, { top: "-100%", opacity: 0 });
}

// Add pulse effect to capture button
gsap.to(captureBtn, { duration: 1, scale: 1.05, repeat: -1, yoyo: true, ease: "power1.inOut" });

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    const userIdInput = document.getElementById('user-id');
    const videoInput = document.getElementById('video-upload');

    if (!userIdInput || !videoInput || !videoInput.files[0]) {
        statusDiv.innerHTML = '<i class="fas fa-exclamation-circle text-red-500"></i> Please provide both user ID and video';
        return;
    }

    formData.append('user_id', userIdInput.value);
    formData.append('video', videoInput.files[0]);

    // Log FormData contents
    for (let [key, value] of formData.entries()) {
        console.log(key, value);
    }

    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';

    try {
        const response = await fetch('/submitAccount', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server response was not ok: ${response.status} ${response.statusText}\n${errorText}`);
        }

        const result = await response.json();
        statusDiv.innerHTML = `<i class="fas fa-check-circle text-green-500"></i> ${result.message}`;
    } catch (error) {
        console.error('Error:', error);
        statusDiv.innerHTML = '<i class="fas fa-exclamation-circle text-red-500"></i> Error submitting form';
    }
});