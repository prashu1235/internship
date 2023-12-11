{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bda9390-fd96-462d-a736-e9cc791c33e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from model_service import ModelService\n",
    "\n",
    "app = Flask(__name__)\n",
    "model_service = ModelService()\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return \"Machine Learning Model Deployment with Flask\"\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        # Get input data from request\n",
    "        data = request.json['data']\n",
    "\n",
    "        # Use the ModelService to get predictions\n",
    "        predictions = model_service.predict(data)\n",
    "\n",
    "        # Return the predictions as JSON\n",
    "        return jsonify(predictions)\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)}), 500\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5000)\n"
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
