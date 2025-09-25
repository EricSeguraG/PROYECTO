import pandas as pd
from flask import Flask, jsonify, request, abort
import os


# Función mejorada para cargar el CSV con validación de archivo
def cargar_csv():
    archivo = "fra_perfumes.csv"
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"El archivo {archivo} no se encuentra en el directorio actual")

    # Lista de posibles configuraciones para cargar el CSV
    configuraciones = [
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1", "quotechar": '"'},
        {"sep": "\t", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"}
    ]

    for i, config in enumerate(configuraciones):
        try:
            return pd.read_csv(archivo, **config)
        except Exception as e:
            print(f"Intento {i + 1} fallido: {e}")
            continue

    raise Exception("No se pudo cargar el CSV con ninguna configuración probada")


# Cargar el DataFrame
try:
    df = cargar_csv()
    print("CSV cargado exitosamente")
    print(f"Dimensiones del DataFrame: {df.shape}")
except Exception as e:
    print(f"Error crítico al cargar el CSV: {e}")
    exit(1)

# Crear columna combinada de main_accords de forma más robusta
main_cols = [c for c in df.columns if c.lower().startswith("mainaccord")]
if main_cols:
    df['main_accords'] = df[main_cols].apply(
        lambda row: [str(v) for v in row if pd.notna(v) and str(v).strip() != ''],
        axis=1
    )
else:
    df['main_accords'] = [[] for _ in range(len(df))]

# Campos a exponer en la API (usando los nombres exactos de las columnas)
CAMPOS_VALIDOS = [
    'url', 'perfume', 'marca', 'genero', 'año', 'salida',
    'corazon', 'base', 'perfumista', 'perfumista 2', 'main_accords'
]

# Solo incluir columnas que realmente existen en el DataFrame
CAMPOS_DISPONIBLES = [campo for campo in CAMPOS_VALIDOS if campo in df.columns]

print("Columnas disponibles en la API:", CAMPOS_DISPONIBLES)

# Crear la aplicación Flask
app = Flask(__name__)


# Función auxiliar para filtrar columnas
def filtrar_campos(df):
    return df[CAMPOS_DISPONIBLES]


# Endpoint: listar todos los perfumes con paginación
@app.route('/perfumes', methods=['GET'])
def get_perfumes():
    try:
        # Parámetros de paginación
        pagina = int(request.args.get('pagina', 1))
        por_pagina = int(request.args.get('por_pagina', 50))

        if pagina < 1 or por_pagina < 1:
            abort(400, description="Los parámetros de paginación deben ser positivos")

        inicio = (pagina - 1) * por_pagina
        fin = inicio + por_pagina

        subset = df.iloc[inicio:fin]
        perfumes = filtrar_campos(subset).to_dict(orient='records')

        return jsonify({
            'pagina': pagina,
            'por_pagina': por_pagina,
            'total': len(df),
            'perfumes': perfumes
        })
    except ValueError:
        abort(400, description="Parámetros de paginación inválidos")


# Endpoint: obtener un perfume por ID
@app.route('/perfumes/<int:perfume_id>', methods=['GET'])
def get_perfume(perfume_id):
    if perfume_id < 0 or perfume_id >= len(df):
        abort(404, description=f"Perfume ID {perfume_id} no encontrado. El rango válido es 0-{len(df) - 1}")

    perfume = filtrar_campos(df.iloc[[perfume_id]]).iloc[0].to_dict()
    return jsonify(perfume)


# Endpoint: búsqueda avanzada con múltiples filtros
@app.route('/perfumes/search', methods=['GET'])
def search_perfumes():
    try:
        query = df.copy()

        # Filtros básicos de texto - USANDO NOMBRES EXACTOS
        filtros_texto = {
            'marca': 'marca',
            'genero': 'genero',
            'perfume': 'perfume',
            'perfumista': 'perfumista',
            'año': 'año'
        }

        for param, columna in filtros_texto.items():
            valor = request.args.get(param)
            if valor and columna in query.columns:
                query = query[query[columna].astype(str).str.contains(valor, case=False, na=False)]

        # BÚSQUEDA DE NOTAS EN CUALQUIER CAMPO DE NOTAS
        nota = request.args.get('nota')
        if nota:
            # Campos donde buscar las notas
            campos_notas = ['salida', 'corazon', 'base', 'main_accords']
            campos_disponibles = [campo for campo in campos_notas if campo in query.columns]

            # Crear una máscara combinada para todos los campos de notas
            mascara_notas = pd.Series([False] * len(query))

            for campo in campos_disponibles:
                if campo == 'main_accords':
                    # Búsqueda especial para la lista de acordes
                    mascara_campo = query[campo].apply(
                        lambda acordes: any(nota.lower() in str(a).lower() for a in acordes)
                    )
                else:
                    # Búsqueda en campos de texto normales
                    mascara_campo = query[campo].astype(str).str.contains(nota, case=False, na=False)

                mascara_notas = mascara_notas | mascara_campo

            query = query[mascara_notas]

        # Filtro por acordes (búsqueda en lista)
        acorde = request.args.get('acorde')
        if acorde and 'main_accords' in query.columns:
            query = query[query['main_accords'].apply(
                lambda acordes: any(acorde.lower() in str(a).lower() for a in acordes)
            )]

        # Ordenamiento
        orden = request.args.get('orden')
        if orden and orden in query.columns:
            ascendente = not request.args.get('desc', '').lower() == 'true'
            query = query.sort_values(by=orden, ascending=ascendente)

        resultados = filtrar_campos(query).to_dict(orient='records')
        return jsonify({
            'total_resultados': len(resultados),
            'parametros_busqueda': {
                'nota': nota,
                'acorde': acorde,
                'marca': request.args.get('marca'),
                'genero': request.args.get('genero'),
                'perfume': request.args.get('perfume'),
                'año': request.args.get('año')
            },
            'resultados': resultados
        })

    except Exception as e:
        abort(500, description=f"Error interno en la búsqueda: {str(e)}")


# Manejador de errores personalizado
@app.errorhandler(404)
def no_encontrado(error):
    return jsonify({'error': str(error)}), 404


@app.errorhandler(400)
def solicitud_incorrecta(error):
    return jsonify({'error': str(error)}), 400


@app.errorhandler(500)
def error_interno(error):
    return jsonify({'error': str(error)}), 500


# Esto debe estar al final del archivo
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)