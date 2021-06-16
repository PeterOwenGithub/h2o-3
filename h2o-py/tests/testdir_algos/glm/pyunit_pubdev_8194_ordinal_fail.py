from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# test taken from Ben Epstein.  Thank you.
def test_model():
    cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")
    cars["cylinders"] = cars["cylinders"].asfactor()
    cars.rename(columns={"year": "year_make"})
    r = cars[0].runif()
    train = cars[r > 0.2]
    valid = cars[r <= 0.2]
    response = "cylinders"
    predictors = [
        "displacement",
        "power",
        "weight",
        "acceleration",
        "year_make",
    ]
    
    model = H2OGeneralizedLinearEstimator(seed=1234, family="ordinal")
    model.train(
        x=predictors, y=response, training_frame=train, validation_frame=valid
    )

    features = h2o.H2OFrame(pd.DataFrame([[18,101,22,23.142,1]], columns=predictors))
    
    model_raw_preds = model.predict(features).as_data_frame().values.tolist()[0]
    
    model_pred = model_raw_preds[0] # Label
    probs = model_raw_preds[1:] # Probabilities
    labels = [3, 4, 5, 6, 8]

    
    max_prob = max(probs)
    max_prob_index = probs.index(max_prob)
    prob_pred = labels[max_prob_index]
    
    label_probs = dict(zip(labels, probs))
    
    print(f'Model pred: {model_pred}, probabilities: {label_probs}')

    assert prob_pred==model_pred, f'Predictions are wrong, model gave {model_pred} but max prob was {prob_pred} with probability {max_prob}. All probs: {label_probs}'

def test_ordinal():
    for _ in range(100):
        test_model()


if __name__ == "__main__":
  pyunit_utils.standalone_test(test_ordinal)
else:
    test_ordinal()
