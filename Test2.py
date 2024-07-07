import unittest
import pandas as pd
from pycaret.datasets import get_data
from AutoML2 import MLSystem  # Asumiendo que la clase MLSystem está en un archivo llamado AutoML2.py

class TestMLSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Carga un conjunto de datos de prueba
        cls.data = pd.read_csv('C:\\Users\\sagit\\Downloads\\train.csv')
        cls.target = 'Target'
        cls.ml_system = MLSystem(cls.data, cls.target)
        cls.ml_system.setup_system()
        cls.ml_system.compare_models()

    # Definir la función get_model_name dentro de la clase TestMLSystem
    @staticmethod
    def get_model_name(e):
        mn = str(e).split("(")[0]

        if 'catboost' in str(e):
            mn = 'CatBoostClassifier'

        model_dict_logging = {'ExtraTreesClassifier': 'Extra Trees Classifier',
                              'GradientBoostingClassifier': 'Gradient Boosting Classifier',
                              'RandomForestClassifier': 'Random Forest Classifier',
                              'LGBMClassifier': 'Light Gradient Boosting Machine',
                              'XGBClassifier': 'Extreme Gradient Boosting',
                              'AdaBoostClassifier': 'Ada Boost Classifier',
                              'DecisionTreeClassifier': 'Decision Tree Classifier',
                              'RidgeClassifier': 'Ridge Classifier',
                              'LogisticRegression': 'Logistic Regression',
                              'KNeighborsClassifier': 'K Neighbors Classifier',
                              'GaussianNB': 'Naive Bayes',
                              'SGDClassifier': 'SVM - Linear Kernel',
                              'SVC': 'SVM - Radial Kernel',
                              'GaussianProcessClassifier': 'Gaussian Process Classifier',
                              'MLPClassifier': 'MLP Classifier',
                              'QuadraticDiscriminantAnalysis': 'Quadratic Discriminant Analysis',
                              'LinearDiscriminantAnalysis': 'Linear Discriminant Analysis',
                              'CatBoostClassifier': 'CatBoost Classifier',
                              'BaggingClassifier': 'Bagging Classifier',
                              'VotingClassifier': 'Voting Classifier'}

        return model_dict_logging.get(mn)

    def test_best_model_is_lightgbm(self):
        # Obtener el mejor modelo seleccionado por PyCaret
        best_model = self.ml_system.best_model

        print(best_model)  # Salida esperada: 'Light Gradient Boosting Machine'

        # Verificar si el nombre del mejor modelo es 'Light Gradient Boosting Machine'
        if self.get_model_name(best_model) == 'Light Gradient Boosting Machine':
            # Si el nombre del modelo es correcto, la prueba pasa automáticamente
            pass
        else:
            self.fail(f"El mejor modelo seleccionado no es Light Gradient Boosting Machine, detalles: {best_model}")



if __name__ == '__main__':
    unittest.main()
