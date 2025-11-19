import pandas as pd
from flask import Flask, jsonify, request, abort
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import math

# --------------------------------------------------
# Crear la aplicación Flask con CORS
# --------------------------------------------------
app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Función para limpiar valores NaN/None antes de convertir a JSON
# --------------------------------------------------
def limpiar_para_json(obj):
    """Convierte valores NaN/None a strings vacíos para JSON válido"""
    if isinstance(obj, dict):
        return {k: limpiar_para_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [limpiar_para_json(v) for v in obj]
    elif pd.isna(obj) or obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return ""
    elif isinstance(obj, (int, float)):
        # Mantener números válidos
        return obj
    else:
        return obj


# --------------------------------------------------
# Función para cargar el CSV
# --------------------------------------------------
def cargar_csv():
    archivo = "fra_perfumes.csv"
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"El archivo {archivo} no se encuentra en el directorio actual")

    configuraciones = [
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1", "quotechar": '"'},
        {"sep": "\t", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"}
    ]

    for i, config in enumerate(configuraciones):
        try:
            df = pd.read_csv(archivo, **config)
            # Limpiar NaN/None del DataFrame inmediatamente
            df = df.fillna("")
            return df
        except Exception as e:
            print(f"Intento {i + 1} fallido: {e}")
            continue

    raise Exception("No se pudo cargar el CSV con ninguna configuración probada")


# --------------------------------------------------
# Cargar el DataFrame
# --------------------------------------------------
try:
    df = cargar_csv()
    print("CSV cargado exitosamente")
    print(f"Dimensiones del DataFrame: {df.shape}")
except Exception as e:
    print(f"Error crítico al cargar el CSV: {e}")
    exit(1)

# Crear columna combinada de main_accords
main_cols = [c for c in df.columns if c.lower().startswith("mainaccord")]
if main_cols:
    df['main_accords'] = df[main_cols].apply(
        lambda row: [str(v) for v in row if pd.notna(v) and str(v).strip() != ''],
        axis=1
    )
else:
    df['main_accords'] = [[] for _ in range(len(df))]

# Campos válidos para exponer en la API
CAMPOS_VALIDOS = [
    'url', 'perfume', 'marca', 'genero', 'año', 'salida',
    'corazon', 'base', 'perfumista', 'perfumista 2', 'main_accords'
]
CAMPOS_DISPONIBLES = [campo for campo in CAMPOS_VALIDOS if campo in df.columns]
print("Columnas disponibles en la API:", CAMPOS_DISPONIBLES)


# --------------------------------------------------
# Preparación de datos
# --------------------------------------------------
def filtrar_campos(df_sub):
    return df_sub[CAMPOS_DISPONIBLES]


def extraer_notas(row):
    notas = []
    for campo in ['salida', 'corazon', 'base']:
        if campo in df.columns and pd.notna(row.get(campo, None)):
            notas += [n.strip().lower() for n in str(row[campo]).split(',')]
    if 'main_accords' in row and isinstance(row['main_accords'], list):
        notas += [str(n).lower() for n in row['main_accords']]
    return list(set(notas))


df['todas_notas'] = df.apply(extraer_notas, axis=1)

# Vocabulario global
VOCAB = sorted({n for notas in df['todas_notas'] for n in notas})


def vectorizar_notas(notas, vocab):
    return [1 if n in notas else 0 for n in vocab]


MATRIZ_VECTORES = np.array([vectorizar_notas(notas, VOCAB) for notas in df['todas_notas']])


# --------------------------------------------------
# Endpoints CORREGIDOS
# --------------------------------------------------
@app.route('/perfumes', methods=['GET'])
def get_perfumes():
    try:
        pagina = int(request.args.get('pagina', 1))
        por_pagina = int(request.args.get('por_pagina', 50))

        if pagina < 1 or por_pagina < 1:
            abort(400, description="Los parámetros de paginación deben ser positivos")

        inicio = (pagina - 1) * por_pagina
        fin = inicio + por_pagina

        subset = df.iloc[inicio:fin]
        perfumes = filtrar_campos(subset).to_dict(orient='records')

        # LIMPIAR para JSON válido
        perfumes_limpios = [limpiar_para_json(perfume) for perfume in perfumes]

        return jsonify({
            'pagina': pagina,
            'por_pagina': por_pagina,
            'total': len(df),
            'perfumes': perfumes_limpios
        })
    except ValueError:
        abort(400, description="Parámetros de paginación inválidos")


@app.route('/perfumes/<int:perfume_id>', methods=['GET'])
def get_perfume(perfume_id):
    if perfume_id < 0 or perfume_id >= len(df):
        abort(404, description=f"Perfume ID {perfume_id} no encontrado. El rango válido es 0-{len(df) - 1}")
    perfume = filtrar_campos(df.iloc[[perfume_id]]).iloc[0].to_dict()
    perfume_limpio = limpiar_para_json(perfume)
    return jsonify(perfume_limpio)


@app.route('/perfumes/search', methods=['GET'])
def search_perfumes():
    try:
        query = df.copy()
        filtros_texto = {
            'marca': 'marca',
            'genero': 'genero',
            'perfume': 'perfume',
            'perfumista': 'perfumista',
            'año': 'año'
        }

        # --- Filtros básicos ---
        for param, columna in filtros_texto.items():
            valor = request.args.get(param)
            if valor and columna in query.columns:
                if columna == 'genero':
                    query = query[query[columna].astype(str).str.lower() == valor.lower()]
                else:
                    query = query[query[columna].astype(str).str.contains(valor, case=False, na=False)]

        # --- Buscar por notas ---
        notas_param = request.args.get('nota')
        if notas_param:
            notas_buscar = [n.strip().lower() for n in notas_param.split(",") if n.strip()]

            def contiene_todas(row):
                notas_perfume = extraer_notas(row)
                return all(n in notas_perfume for n in notas_buscar)

            query = query[query.apply(contiene_todas, axis=1)]

        # --- Buscar por acordes ---
        acordes_param = request.args.get('acorde')
        if acordes_param and 'main_accords' in query.columns:
            acordes_buscar = [a.strip().lower() for a in acordes_param.split(",") if a.strip()]

            def contiene_todos_acordes(acordes):
                acordes_lower = [str(a).lower() for a in acordes]
                return all(a in acordes_lower for a in acordes_buscar)

            query = query[query['main_accords'].apply(contiene_todos_acordes)]

        # --- Ordenar resultados ---
        orden = request.args.get('orden')
        if orden and orden in query.columns:
            ascendente = not request.args.get('desc', '').lower() == 'true'
            query = query.sort_values(by=orden, ascending=ascendente)

        resultados = filtrar_campos(query).to_dict(orient='records')
        resultados_limpios = [limpiar_para_json(resultado) for resultado in resultados]

        return jsonify({
            'total_resultados': len(resultados_limpios),
            'parametros_busqueda': {
                'nota': notas_param,
                'acorde': acordes_param,
                'marca': request.args.get('marca'),
                'genero': request.args.get('genero'),
                'perfume': request.args.get('perfume'),
                'año': request.args.get('año')
            },
            'resultados': resultados_limpios
        })

    except Exception as e:
        abort(500, description=f"Error interno en la búsqueda: {str(e)}")


@app.route('/perfumes/similares', methods=['GET'])
def get_similares_nombre():
    nombre = request.args.get('nombre')
    if not nombre:
        abort(400, description="Debes proporcionar el parámetro 'nombre'")

    coincidencias = df[df['perfume'].astype(str).str.contains(nombre, case=False, na=False)]
    if coincidencias.empty:
        abort(404, description=f"No se encontró ningún perfume que coincida con '{nombre}'")

    idx_base = coincidencias.index[0]
    base_vec = MATRIZ_VECTORES[idx_base].reshape(1, -1)

    similitudes = cosine_similarity(base_vec, MATRIZ_VECTORES)[0]
    df['score_similaridad'] = similitudes

    similares = df[df.index != idx_base].sort_values('score_similaridad', ascending=False)
    top_n = int(request.args.get('n', 10))
    similares = similares.head(top_n)

    similares_out = filtrar_campos(similares).copy()
    similares_out['similitud'] = (similares['score_similaridad'] * 100).round(1).astype(str)

    # LIMPIAR para JSON válido
    base_limpio = limpiar_para_json(filtrar_campos(df.iloc[[idx_base]]).iloc[0].to_dict())
    similares_limpios = [limpiar_para_json(perfume) for perfume in similares_out.to_dict(orient='records')]

    return jsonify({
        'base': base_limpio,
        'similares': similares_limpios
    })


# --------------------------------------------------
# Manejadores de error
# --------------------------------------------------
@app.errorhandler(404)
def no_encontrado(error):
    return jsonify({'error': str(error)}), 404


@app.errorhandler(400)
def solicitud_incorrecta(error):
    return jsonify({'error': str(error)}), 400


@app.errorhandler(500)
def error_interno(error):
    return jsonify({'error': str(error)}), 500


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)