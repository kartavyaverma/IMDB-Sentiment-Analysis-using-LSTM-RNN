# IMDB Movie Review Sentiment Analysis

This is an end-to-end Machine Learning project that classifies movie reviews as either Positive or Negative. 

It uses a Recurrent Neural Network (an LSTM) built with TensorFlow/Keras, and features a clean, responsive front-end graphical user interface built with Streamlit.

## Files In This Repository

* `main.py` - The Streamlit graphical web application and inference logic.
* `train_lstm.py` - The standalone Python script containing the training code. Use this to download the dataset and train the model from scratch.
* `requirements.txt` - The Python library dependencies.

> **Note:** The pre-trained weights for the LSTM model (`lstm_imdb.h5`) are large (~70MB) and are intentionally excluded from this repository. You will need to run the training script to generate the model before starting the web app.

## How to Run the Web App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kartavyaverma/IMDB-Sentiment-Analysis-using-LSTM-RNN.git
   cd IMDB-Sentiment-Analysis-using-LSTM-RNN
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model & Generate weights:**
   Because the model file (`lstm_imdb.h5`) is not included in the repository, you must generate it first.
   ```bash
   python train_lstm.py
   ```
   *This will download the IMDB dataset, train the LSTM over a few epochs, and save `lstm_imdb.h5` in your project folder.*

5. **Launch the web application:**
   ```bash
   streamlit run main.py
   ```
   *The application will automatically open in your web browser at `http://localhost:8501`*

## Tweaking The Model
If you want to tweak the LSTM's architecture:
1. Open `train_lstm.py` in your favorite IDE to adjust parameters like epochs, vocabulary size, batch size, or LSTM structure.
2. Run the file to save the new `lstm_imdb.h5` file.
3. The web app will immediately start using your newly trained model on the next run!
