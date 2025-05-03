from src.predict import predict

def test_prediction():
    assert predict([1, 2]) in [0, 1]