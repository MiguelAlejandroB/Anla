import numpy as np #Importo las librerias que voy a usar fuera de la clase
import pandas as pd
class Anla:
    df = pd.read_csv("est_25027400_limpia.csv")
    # Establecer la primera columna como el índice del DataFrame
    df.set_index(df.columns[0], inplace=True)

    # Convertir el índice a tipo de datos de fecha
    df.index = pd.to_datetime(df.index)
    # Renombrar la segunda columna como 'Caudal'
    # Renombrar la segunda columna como 'Caudal'
    df.rename(columns={1: 'Caudal'}, inplace=True)

    # Calcular el promedio de los caudales cada 7 días
    promedio_7_dias = df.resample('7D').mean()
    