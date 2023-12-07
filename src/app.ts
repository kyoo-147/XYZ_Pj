import * as faceapi from 'face-api.js';

document.addEventListener('DOMContentLoaded', async () => {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

    navigator.getUserMedia(
        { video: {} },
        async (stream) => {
            const video = document.getElementById('video') as HTMLVideoElement;
            video.srcObject = stream;

            video.addEventListener('play', () => {
                const overlay = document.getElementById('overlay') as HTMLCanvasElement;
                const displaySize = { width: video.width, height: video.height };
                faceapi.matchDimensions(overlay, displaySize);

                setInterval(async () => {
                    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                        .withFaceLandmarks()
                        .withFaceDescriptors();

                    const resizedDetections = faceapi.resizeResults(detections, displaySize);
                    overlay.getContext('2d').clearRect(0, 0, displaySize.width, displaySize.height);
                    faceapi.draw.drawDetections(overlay, resizedDetections);
                    faceapi.draw.drawFaceLandmarks(overlay, resizedDetections);
                }, 100);
            });
        },
        (err) => console.error(err)
    );
});
