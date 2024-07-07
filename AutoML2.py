import pandas as pd
from pycaret.classification import setup, compare_models, create_model, tune_model, evaluate_model, predict_model, save_model, load_model

class MLSystem:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.model = None
        self.setup_data = None
        self.best_model = None

    def setup_system(self):
        # Configura el entorno de PyCaret
        self.setup_data = setup(data=self.data, target=self.target)

    def compare_models(self):
        # Compara diferentes modelos y elige el mejor
        self.best_model = compare_models()

    def create_model(self):
        # Crea el modelo con mayor precisión
        if self.best_model is not None:
            self.model = create_model(self.best_model)

    def tune_model(self):
        # Ajusta el modelo creado
        if self.model is not None:
            self.model = tune_model(self.model)

    def evaluate_model(self):
        # Evalúa el modelo ajustado
        if self.model is not None:
            evaluate_model(self.model)

    def predict(self, data: pd.DataFrame):
        # Realiza predicciones con el modelo ajustado
        if self.model is not None:
            predictions = predict_model(self.model, data=data)
            return predictions
        else:
            raise Exception("No model has been created yet.")

    def save_model(self, model_name: str):
        # Guarda el modelo en un archivo
        if self.model is not None:
            save_model(self.model, model_name)
        else:
            raise Exception("No model has been created yet.")

    def load_model(self, model_name: str):
        # Carga un modelo desde un archivo
        self.model = load_model(model_name)

def main(data_path: str, target: str, new_data_path: str):
    # Carga tus datos aquí
    data = pd.read_csv(data_path)

    ml_system = MLSystem(data, target)
    ml_system.setup_system()
    ml_system.compare_models()
    ml_system.create_model()
    ml_system.tune_model()
    ml_system.evaluate_model()

    # Guardar el modelo
    ml_system.save_model('best_model')

    # Para predecir con nuevos datos
    new_data = pd.read_csv(new_data_path)
    predictions = ml_system.predict(new_data)
    print(predictions)

if __name__ == "__main__":
    # Definir las rutas de los archivos y el nombre del modelo
    data_path = 'C:\\Users\\sagit\\Downloads\\train.csv'
    target_column = 'Target'
    new_data_path = 'C:\\Users\\sagit\\Downloads\\test.csv'

    # Ejecutar la función principal
    main(data_path, target_column, new_data_path)
