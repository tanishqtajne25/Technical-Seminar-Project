# LipBuddy (LipNet Full Stack App)

LipBuddy is an end-to-end deep learning application based on the famous [LipNet](https://arxiv.org/abs/1611.01599) architecture. It takes a silent video of a person speaking and uses a **3D Convolutional Neural Network + Bidirectional LSTM** to transcribe their lip movements into text.

The application is bundled with a sleek, interactive frontend powered by **Streamlit**.

---

## 📂 Project Structure

- `app/` - Contains the Streamlit frontend and inference code
  - `streamlitapp.py` - The main web UI script
  - `modelutil.py` - Defines and loads the LipNet neural network architecture
  - `utils.py` - Handles video loading, frame normalization, and CTC alignment parsing
- `LipNet.ipynb` - The original model training and experimentation notebook
- `models - checkpoint 96/` - Saved pre-trained weights for the model (Epoch 96)
- `requirements.txt` - Python dependencies needed to run the UI

---

## 📀 Data Requirements

This project is trained on the **GRID Audio-Visual Sentence Corpus**.

By default, the app expects your data to be located at:
`D:\TS\ypc\data`

Inside that folder, videos must be organized by speaker directories with an `align/` subfolder, like so:

```text
D:\TS\ypc\data\
├── s1_processed\
│   ├── bbaf2n.mpg          <-- Real-time Video
│   ├── bbal6n.mpg
│   └── align\              <-- Text Annotations (Ground truth)
│       ├── bbaf2n.align
│       └── bbal6n.align
├── s2_processed\
│   └── ...
```

### 🔄 Changing the Data File Path

If your GRID Corpus data is located somewhere else, you must update the path in the code before running the app.

1. Open `app/utils.py`.
2. Locate the `DATA_ROOT` variable near the top of the file.
3. Change it to point to your new absolute data path:
   ```python
   # Root path to the GRID corpus data
   DATA_ROOT = r'C:\Path\To\Your\Data'
   ```
   _(Make sure to keep the `r` before the string if you are on Windows to handle backslashes correctly!)_

---

## ⚙️ Installation & Setup

1. **Clone or Download the Repository**.
2. **Create a Virtual Environment** (We highly recommend using Python 3.10 / 3.11):

   ```bash
   python -m venv venv
   ```

   or

   ```
   conda create -p venv python=3.11 -y // for conda users
   ```

3. **Activate the Environment**:
   - **Windows (Command Prompt / PowerShell)**:
     ```powershell
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
4. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> 💡 **Note on Video Conversion**: Streamlit needs the video to be converted from `.mpg` to `.mp4` for the browser. The `imageio-ffmpeg` package is included in the requirements to handle this internally without requiring a global `ffmpeg` installation.

---

## 🚀 How to Run the App

Once your data is present and your environment is active, start the Streamlit web server:

```bash
cd app
streamlit run streamlitapp.py
```

- A new tab will automatically open in your default browser (usually at `http://localhost:8501`).
- **Select a Speaker** from the sidebar/dropdown.
- **Select a Video** to run inference on.
- The model will convert the `.mpg` video into an `.mp4` preview, extract the grayscale crop of the lips as an animated `.gif`, and then output the decoded text predictions in real-time!

---

## 🧠 Model Architecture

The LipBuddy architecture consists of three main components:

1. **Spatial Feature Extraction (3D CNN)**
   - Three cascaded `Conv3D` layers (128, 256, and 75 filters respectively).
   - Each Convolution is followed by a `ReLU` activation and a `MaxPool3D` layer for spatial downsampling.
   - This part of the network focuses on tracking the physical movements and shapes of the lips across the frame sequence.

2. **Temporal Sequence Modelling (Bi-LSTM)**
   - The spatial features are flattened via `TimeDistributed(Flatten())`.
   - Passed into two `Bidirectional(LSTM)` layers (128 units each) to understand the *context* of the movement over time (both forwards and backwards).
   - `Dropout(0.5)` is applied between the recurrent layers for regularization.

3. **Classification & CTC Decode**
   - A final `Dense` layer with a `softmax` activation outputs character probabilities for each timeframe into 41 classes (the vocabulary + CTC blank tokens).
   - We use **CTC (Connectionist Temporal Classification)** to decode the output sequence, collapsing repeated characters and removing blanks to form the final transcribed words!
