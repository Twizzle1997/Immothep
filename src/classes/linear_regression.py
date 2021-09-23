from classes.predictor import Predictor
from sklearn.linear_model import LinearRegression

class Linear_reg(Predictor):

    def __init__(self):
        self.model = LinearRegression(n_jobs = 10, positive=True)
                                