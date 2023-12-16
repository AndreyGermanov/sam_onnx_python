import onnxruntime as ort
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import numpy as np
from copy import deepcopy

app = Flask(__name__)

orig_width, orig_height = [0, 0]
resized_width, resized_height = [0, 0]
image_embeddings = None


def main():
    serve(app, host='0.0.0.0', port=8080)


@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()


@app.route("/encode", methods=["POST"])
def encode():
    buf = request.files["image_file"]
    create_image_embeddings(buf)
    return '{"status":"ok"}'


def create_image_embeddings(buf):
    global image_embeddings
    input_tensor = prepare_input_tensor(buf)
    encoder = ort.InferenceSession("vit_b_encoder.onnx")
    outputs = encoder.run(None, {"images": input_tensor})
    image_embeddings = outputs[0]


def prepare_input_tensor(buf):
    global orig_width, orig_height, resized_width, resized_height

    # Open image from uploaded file buffer
    img = Image.open(buf)
    img = img.convert("RGB")

    # Resize image to 1024 preserving aspect ratio
    orig_width, orig_height = img.size
    resized_width, resized_height = [0, 0]

    if orig_width > orig_height:
        resized_width = 1024
        resized_height = int(1024 / orig_width * orig_height)
    else:
        resized_height = 1024
        resized_width = int(1024 / orig_height * orig_width)

    img = img.resize((resized_width, resized_height), Image.BILINEAR)
    input_tensor = np.array(img)

    # Normalize input tensor numbers
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([[58.395, 57.12, 57.375]])
    input_tensor = (input_tensor - mean) / std

    # Transpose input tensor to shape (Batch,Channels,Height,Width)
    input_tensor = input_tensor.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    # Pad input tensor by zeros to 1024x1024
    if resized_height < resized_width:
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 1024 - resized_height), (0, 0)))
    else:
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (0, 1024 - resized_width)))

    return input_tensor


@app.route("/decode", methods=["POST"])
def decode():
    box = [int(item) for item in request.form["box"].split(",")]
    prompt = encode_prompt(box)
    mask = decode_mask(box, prompt)
    return jsonify(mask)


def encode_prompt(box):
    input_box = np.array(box).reshape(2, 2)
    input_labels = np.array([2, 3])
    onnx_coord = input_box[None, :, :]
    onnx_label = input_labels[None, :].astype(np.float32)
    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)
    onnx_coord = coords.astype("float32")
    return onnx_coord, onnx_label


def decode_mask(box, prompt):
    onnx_coord, onnx_label = prompt
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    decoder = ort.InferenceSession("vit_b_decoder.onnx")
    masks, _, _ = decoder.run(None, {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array([orig_height, orig_width], dtype=np.float32)
    })
    mask = (masks[0][0] > 0)
    x1, y1, x2, y2 = box
    return mask[y1:y2, x1:x2].flatten().tolist()


main()
