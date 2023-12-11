{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9481ed16-38b7-4532-b4d2-333f3d908784",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting flask\n",
      "  Downloading flask-3.0.0-py3-none-any.whl.metadata (3.6 kB)\n",
      "Collecting Werkzeug>=3.0.0 (from flask)\n",
      "  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)\n",
      "Requirement already satisfied: Jinja2>=3.1.2 in /opt/conda/lib/python3.11/site-packages (from flask) (3.1.2)\n",
      "Collecting itsdangerous>=2.1.2 (from flask)\n",
      "  Downloading itsdangerous-2.1.2-py3-none-any.whl (15 kB)\n",
      "Requirement already satisfied: click>=8.1.3 in /opt/conda/lib/python3.11/site-packages (from flask) (8.1.7)\n",
      "Requirement already satisfied: blinker>=1.6.2 in /opt/conda/lib/python3.11/site-packages (from flask) (1.7.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.11/site-packages (from Jinja2>=3.1.2->flask) (2.1.3)\n",
      "Downloading flask-3.0.0-py3-none-any.whl (99 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m99.7/99.7 kB\u001b[0m \u001b[31m583.4 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
      "\u001b[?25hDownloading werkzeug-3.0.1-py3-none-any.whl (226 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m226.7/226.7 kB\u001b[0m \u001b[31m937.9 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
      "\u001b[?25hInstalling collected packages: Werkzeug, itsdangerous, flask\n",
      "Successfully installed Werkzeug-3.0.1 flask-3.0.0 itsdangerous-2.1.2\n"
     ]
    }
   ],
   "source": [
    "!pip install flask\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2353338-cd36-4b2d-afb1-ed423bf28ef6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel, LogisticRegressionModel\n",
    "from pyspark.ml.clustering import KMeansModel\n",
    "from pyspark.ml.linalg import Vectors\n",
    "\n",
    "class ModelService:\n",
    "    def __init__(self):\n",
    "        # Load the machine learning models\n",
    "        self.rf_model = RandomForestClassificationModel.load(\"random_forest_model\")\n",
    "        self.dt_model = DecisionTreeClassificationModel.load(\"decision_tree_model\")\n",
    "        self.lr_model = LogisticRegressionModel.load(\"logistic_regression_model\")\n",
    "        self.kmeans_model = KMeansModel.load(\"kmeans_model\")\n",
    "\n",
    "    def predict(self, data):\n",
    "        # Prepare data for prediction (adjust this based on your model features)\n",
    "        features = Vectors.dense(data['feature1'], data['feature2'], data['feature3'])\n",
    "\n",
    "        # Make predictions using each model\n",
    "        rf_prediction = self.rf_model.transform(features)\n",
    "        dt_prediction = self.dt_model.transform(features)\n",
    "        lr_prediction = self.lr_model.transform(features)\n",
    "        kmeans_prediction = self.kmeans_model.transform(features)\n",
    "\n",
    "        # Extract prediction results (modify this based on your model's output)\n",
    "        rf_result = rf_prediction.select('prediction').collect()[0]['prediction']\n",
    "        dt_result = dt_prediction.select('prediction').collect()[0]['prediction']\n",
    "        lr_result = lr_prediction.select('prediction').collect()[0]['prediction']\n",
    "        kmeans_result = kmeans_prediction.select('prediction').collect()[0]['prediction']\n",
    "\n",
    "        # Return the predictions as a dictionary\n",
    "        return {\n",
    "            'RandomForest': int(rf_result),\n",
    "            'DecisionTree': int(dt_result),\n",
    "            'LogisticRegression': int(lr_result),\n",
    "            'KMeans': int(kmeans_result)\n",
    "        }\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
