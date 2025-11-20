# Author: vlarobbyk (adaptado por Mateo)
# Version: 1.0
# Date: 2024-10-20
# Updated: 2025-11-20

from flask import Flask, render_template, Response, request
from io import BytesIO
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

from background import  remove_background

app = Flask(__name__)


_URL = 'http://192.168.1.2'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])


noise_params = {
    "gauss_mean": 0.0,
    "gauss_std": 20.0,
    "speckle_var": 0.05
}


def video_capture():
    res = requests.get(stream_url, stream=True)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    last_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                h, w = gray.shape

                median_bg, mask = remove_background(gray)

                foreground = cv2.bitwise_and(gray, gray, mask=mask)

                hist = cv2.equalizeHist(gray)

                clahe_img = clahe.apply(gray)

                bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

                now = cv2.getTickCount()
                fps = freq / (now - last_time)
                last_time = now

                panel = np.zeros((h*3, w*3), dtype=np.uint8)

                panel[0:h, 0:w]       = gray
                panel[0:h, w:2*w]     = median_bg
                panel[0:h, 2*w:3*w]   = mask

                panel[h:2*h, 0:w]     = hist
                panel[h:2*h, w:2*w]   = clahe_img
                panel[h:2*h, 2*w:3*w] = bilateral

                panel[2*h:3*h, 0:w]     = foreground
                panel[2*h:3*h, w:2*w]   = np.zeros_like(gray)      
                panel[2*h:3*h, 2*w:3*w] = np.zeros_like(gray)      

                cv2.putText(panel, "Original", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(panel, "Fondo (Mediana)", (w+10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(panel, "Mask", (2*w+10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

                cv2.putText(panel, "HistEq", (10, h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(panel, "CLAHE", (w+10, h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(panel, "Bilateral (Investigado)", (2*w+10, h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

                cv2.putText(panel, "Foreground (bitwise AND)", (10, 2*h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

                cv2.putText(panel, f"FPS: {fps:.2f}", (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

                ok, encoded = cv2.imencode(".jpg", panel)
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    encoded.tobytes() +
                    b"\r\n"
                )

            except Exception as e:
                print("Error:", e)
                continue

def video_noise_capture():
    res = requests.get(stream_url, stream=True)

    global noise_params

    kernel_torch = torch.tensor([
        [0, -1/5, 0],
        [-1/5, 2.2, -1/5],
        [0, -1/5, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    last_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) < 100:
            continue

        try:
            img_data = BytesIO(chunk)
            frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

            if frame is None:
                continue

            h, w, _ = frame.shape


            mean = noise_params["gauss_mean"]
            std  = noise_params["gauss_std"]
            var  = noise_params["speckle_var"]

            gauss_noise = np.random.normal(mean, std, frame.shape).astype(np.int16)
            noisy_gauss = frame.astype(np.int16) + gauss_noise
            noisy_gauss = np.clip(noisy_gauss, 0, 255).astype(np.uint8)

            speckle_noise = np.random.randn(*frame.shape) * var
            noisy_speckle = frame + (frame * speckle_noise)
            noisy_speckle = np.clip(noisy_speckle, 0, 255).astype(np.uint8)


            median5  = cv2.medianBlur(noisy_gauss, 5)
            gblur7   = cv2.GaussianBlur(noisy_gauss, (7,7), 0)
            blur7    = cv2.blur(noisy_gauss, (7,7))


            B, G, R = cv2.split(noisy_gauss)
            filtered_channels = []

            for ch in (B, G, R):
                ch_torch = torch.from_numpy(ch.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                ch_out = F.conv2d(ch_torch, kernel_torch, padding=1)
                ch_out = ch_out.squeeze().numpy()
                ch_out = np.clip(ch_out, 0, 255).astype(np.uint8)
                filtered_channels.append(ch_out)

            torch_color = cv2.merge(filtered_channels)


            gray = cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2GRAY)

            canny = cv2.Canny(gray, 80, 160)

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel  = cv2.magnitude(sobelx, sobely)
            sobel  = np.clip(sobel, 0, 255).astype(np.uint8)

            sobel_color = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            canny_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)


            panel = np.zeros((h*3, w*3, 3), dtype=np.uint8)

            panel[0:h,     0:w      ] = frame
            panel[0:h,     w:2*w    ] = noisy_gauss
            panel[0:h,     2*w:3*w  ] = noisy_speckle

            panel[h:2*h,   0:w      ] = median5
            panel[h:2*h,   w:2*w    ] = gblur7
            panel[h:2*h,   2*w:3*w  ] = torch_color

            panel[2*h:3*h, 0:w      ] = canny_color
            panel[2*h:3*h, w:2*w    ] = sobel_color
            panel[2*h:3*h, 2*w:3*w  ] = blur7


            cv2.putText(panel, "Original", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Gauss Noise", (w+10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Speckle Noise", (2*w+10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Median (5x5)", (10, h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Gaussian (7x7)", (w+10, h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "PyTorch Color", (2*w+10, h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Canny", (10, 2*h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Sobel", (w+10, 2*h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(panel, "Blur (7x7)", (2*w+10, 2*h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            now = cv2.getTickCount()
            fps = freq / (now - last_time)
            last_time = now

            cv2.putText(panel, f"FPS: {fps:.2f}", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


            ok, encoded = cv2.imencode(".jpg", panel)
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   encoded.tobytes() +
                   b"\r\n")

        except Exception as e:
            print("Error en video_noise_capture:", e)
            continue


def procesar_morfologia(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    kernels = [
        np.ones((15,15), np.uint8),
        np.ones((25,25), np.uint8),
        np.ones((37,37), np.uint8)
    ]

    resultados = []

    for k in kernels:
        erosion = cv2.erode(img, k)

        dilation = cv2.dilate(img, k)

        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)

        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)

        enhanced = cv2.add(img, (tophat - blackhat))
        

        resultados.append((erosion, dilation, tophat, blackhat, enhanced))

    panel = np.zeros((h*3, w*5), dtype=np.uint8)

    for i in range(3):
        erosion, dilation, tophat, blackhat, enhanced = resultados[i]

        y = i * h

        panel[y:y+h,       0:w]     = erosion
        panel[y:y+h,       w:2*w]   = dilation
        panel[y:y+h,       2*w:3*w] = tophat
        panel[y:y+h,       3*w:4*w] = blackhat
        panel[y:y+h,       4*w:5*w] = enhanced

    return panel





@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(
        video_capture(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/ruido")
def ruido():
    return render_template("noise.html")




@app.route("/update_noise_params", methods=["POST"])
def update_noise_params():
    global noise_params
    data = request.get_json()

    noise_params["gauss_mean"] = float(data["mean"])
    noise_params["gauss_std"]  = float(data["std"])
    noise_params["speckle_var"] = float(data["var"])

    return "OK", 200

@app.route("/video_noise_stream")
def video_noise_stream():
    return Response(video_noise_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/morfologia")
def morfologia():
    imagenes = ["1.jpeg", "2.jpg", "3.jpg"]
    return render_template("morfologia.html", imagenes=imagenes)

@app.route("/morfologia_process/<nombre>")
def morfologia_process(nombre):
    path = f"static/medicas/{nombre}"

    result = procesar_morfologia(path)

    ok, encoded = cv2.imencode(".jpg", result)
    return Response(encoded.tobytes(), mimetype="image/jpeg")




if __name__ == "__main__":
    app.run(debug=False)
