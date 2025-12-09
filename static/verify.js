const video = document.getElementById("video");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Webcam error:", err));

async function captureAndSend() {

    // Create canvas and draw current frame
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0);

    // Convert frame → blob
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));

    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");

    // Send to Flask
    const res = await fetch("/verify", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    console.log("Server response:", data);

    // -------- MATCH BACKEND RESPONSES --------
    if (data.status === "access_granted") {
        window.location = "/success";
    } else if (data.status === "access_denied") {
        window.location = "/denied";
    } else if (data.status === "nohand") {
        document.getElementById("status").innerText = "No hand detected!";
    } else {
        document.getElementById("status").innerText = "Error processing frame!";
    }
}

// Capture every 1 second
setInterval(captureAndSend, 1000);