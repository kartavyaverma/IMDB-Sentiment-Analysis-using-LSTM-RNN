# IMDB Movie Review Sentiment Analysis

This is an end-to-end Machine Learning project that classifies movie reviews as either Positive or Negative. 

It uses a Recurrent Neural Network (an LSTM) built with TensorFlow/Keras, and features a clean, responsive front-end graphical user interface built with Streamlit.

## Files In This Repository

* `main.py` - The Streamlit graphical web application and inference logic.
* `train_lstm.ipynb` - The standalone Jupyter Notebook containing the training code. Use this to re-download the dataset and train the model from scratch.
* `lstm_imdb.h5` - The pre-trained weights for the LSTM model. 
* `requirements.txt` - The Python library dependencies.

## How to Run the Web App Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repo-folder>
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

4. **Launch the app:**
   ```bash
   streamlit run main.py
   ```
   *The application will automatically open in your web browser at `http://localhost:8501`*

## How to Retrain the Model
If you want to train the model from scratch or tweak the LSTM's architecture:
1. Open `train_lstm.ipynb` using Jupyter Notebook or your IDE to adjust parameters like epochs, vocabulary size, or LSTM structure.
2. Run the notebook all the way through to the end.
3. A new `lstm_imdb.h5` file will be generated automatically, which the web app will immediately start using.
