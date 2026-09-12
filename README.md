
# Audio to Text NLP

**Generating the Model Locally**
The trained model file (`twitter_sentiment_model.pkl`) exceeds GitHub's file size limits and is not included. You must generate it locally before running the main application.

**1. Install Dependencies**
Install all required libraries for the data pipeline:
`pip install -r requirements.txt`

**2. Generate the Model**
Open the `Audio to text.ipynb` Jupyter Notebook and run all cells. This will process the provided `twitter.csv` dataset and generate the `twitter_sentiment_model.pkl` file in your root directory.

**3. Start the Application**
Once the pickle file is generated, launch the app:
`python app.py`
