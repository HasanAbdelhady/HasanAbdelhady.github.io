from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from skimage import io as skio
from skimage.transform import resize
import numpy as np
import pickle
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
import cv2
matplotlib.use('Agg')


app = Flask(__name__)

# Load the clustering model


def load_clustering_model():
    with open('colormodel.pkl', 'rb') as model_file:
        return pickle.load(model_file)


# Default K value for the model
default_k = 5  # You can change this to your preferred default K value

clustering_model = load_clustering_model()


def process_image(file, k):
    # Read the uploaded image using the alias skio.imread
    img = skio.imread(file)

    original_image = img.copy()

    img = cv2.resize(img, (50, 50))

    # Reshape the image
    img = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

    # Update the K value of the clustering model
    clustering_model.n_clusters = k

    # Predict cluster labels for each pixel
    cluster_labels = clustering_model.fit_predict(img)

    # Calculate the histogram of cluster labels
    hist, _ = np.histogram(cluster_labels, bins=np.arange(
        0, len(np.unique(cluster_labels)) + 1))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum()

    # Get the RGB values of cluster centers
    cluster_centers = clustering_model.cluster_centers_.astype(int)

    # Calculate the percentage of each color
    percentages = (hist * 100).round(2)

    # Create a list of colors and percentages
    colors_with_percentages = [(f'rgb({c[0]}, {c[1]}, {c[2]})', p) for c, p in zip(
        cluster_centers, percentages)]

    # Rectangle Histogram
    startX = 0
    width = 20  # Adjust the width of each bar as needed

    hist_bar = np.zeros((50, 400, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, clustering_model.cluster_centers_):
        endX = startX + (percent * 400)  # to match grid
        cv2.rectangle(hist_bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # Save the bar chart as a PNG image and encode it in base64
    chart_buffer = io.BytesIO()
    plt.figure(figsize=(6, 2))
    plt.imshow(hist_bar)
    plt.savefig(chart_buffer, format='png', bbox_inches='tight', pad_inches=0)
    chart_buffer.seek(0)
    chart_base64 = base64.b64encode(chart_buffer.read()).decode()
    plt.close()

    # Turning the image back into a numpy array then resizing

    Image.fromarray(original_image, "RGB")
    if original_image.shape[0] >= 1100:
        scale_percent = 50  # percent of original size
    else:
        scale_percent = 60
    width = int(original_image.shape[1] * scale_percent / 100)
    height = int(original_image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    return resized, colors_with_percentages, chart_base64


@app.route('/')
def index():
    return render_template('index.html', result=None, default_k=default_k)


@app.route('/upload', methods=['POST'])
def upload():

    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    # Get K from the form input (default to default_k)
    k = int(request.form.get('k', default_k))

    # Process the uploaded image with the specified K
    img, colors_with_percentages, chart_base64 = process_image(image, k)

    # Convert the processed image to base64 for display in HTML
    img_buffer = io.BytesIO()
    skio.imsave(img_buffer, img, format="JPEG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    # Replace 'result' with the extracted information
    result = {
        'image_base64': img_base64,
        'colors': colors_with_percentages,
        'chart_base64': chart_base64  # Add the chart base64 data
    }

    return render_template('index.html', result=result, default_k=default_k)


if __name__ == '__main__':
    app.run(debug=True)
