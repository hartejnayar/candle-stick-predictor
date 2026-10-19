from src.prediction.predictor import PredictionEngine

def test_prediction_engine_initialization():
    engine = PredictionEngine()
    assert engine is not None