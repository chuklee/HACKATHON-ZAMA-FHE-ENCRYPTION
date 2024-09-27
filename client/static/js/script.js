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
gsap.to(captureBtn, {duration: 1, scale: 1.05, repeat: -1, yoyo: true, ease: "power1.inOut"});

// Handle form submission
form.addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Convert canvas to blob
    canvas.toBlob((blob) => {
        const formData = new FormData(form);
        formData.append('image', blob, 'captured_image.jpg');

        // Send data to server
        fetch('/submit', {
            method: 'POST',
            body: formData
        }).then(response => response.text())
          .then(html => {
              document.body.innerHTML = html;
          }).catch(error => {
              console.error('Error:', error);
          });
    }, 'image/jpeg');
});