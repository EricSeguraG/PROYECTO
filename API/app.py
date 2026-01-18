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
# Función para obtener la ruta absoluta del archivo CSV
# --------------------------------------------------
def obtener_ruta_csv():
    """Obtiene la ruta absoluta del archivo CSV independientemente del directorio de trabajo"""
    # Obtener el directorio donde está este script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Primero intentar en el directorio actual del script
    ruta_csv = os.path.join(directorio_actual, 'fra_perfumes.csv')
    if os.path.exists(ruta_csv):
        return ruta_csv
    
    # Si no está, intentar en el directorio padre
    directorio_padre = os.path.dirname(directorio_actual)
    ruta_csv = os.path.join(directorio_padre, 'fra_perfumes.csv')
    if os.path.exists(ruta_csv):
        return ruta_csv
    
    # Si no está en ninguno de los lugares esperados, devolver la ruta del directorio actual
    return os.path.join(directorio_actual, 'fra_perfumes.csv')


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
# Función para reemplazar guiones por espacios en respuestas
# --------------------------------------------------
def reemplazar_guiones_respuesta(obj):
    """Reemplaza guiones por espacios en strings solo para mostrar en frontend"""
    if isinstance(obj, dict):
        return {k: reemplazar_guiones_respuesta(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [reemplazar_guiones_respuesta(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("-", " ")
    else:
        return obj


# --------------------------------------------------
# Diccionarios de traducción
# --------------------------------------------------
TRADUCCION_NOTAS = {

    # Generales
    'amber': 'ámbar',
    'ambergris': 'ámbar gris',
    'woody': 'madera',
    'citrus': 'cítrico',
    'floral': 'floral',
    'fresh': 'fresco',
    'spicy': 'especiado',
    'aromatic': 'aromático',
    'green': 'verde',
    'fruity': 'frutal',
    'gourmand': 'gourmand',
    'leather': 'cuero',
    'chypre': 'chypre',
    'oriental': 'oriental',
    'powdery': 'polvoso',
    'musky': 'almizclado',
    'aquatic': 'acuático',
    'herbal': 'herbal',

    # Flores
    'rose': 'rosa',
    'jasmine': 'jazmín',
    'orange blossom': 'flor de azahar',
    'tuberose': 'tuberosa',
    'ylang-ylang': 'ylang-ylang',
    'iris': 'iris',
    'violet': 'violeta',
    'lavender': 'lavanda',
    'magnolia': 'magnolia',
    'freesia': 'fresia',
    'lotus': 'loto',
    'heliotrope': 'heliotropo',
    'osmanthus': 'osmanto',
    'lily': 'lirio',

    # Frutas
    'bergamot': 'bergamota',
    'lemon': 'limón',
    'orange': 'naranja',
    'mandarin': 'mandarina',
    'grapefruit': 'pomelo',
    'apple': 'manzana',
    'pear': 'pera',
    'peach': 'durazno',
    'blackcurrant': 'grosella negra',
    'raspberry': 'frambuesa',
    'strawberry': 'fresa',
    'cherry': 'cereza',
    'plum': 'ciruela',
    'fig': 'higo',
    'pineapple': 'piña',
    'melon': 'melón',
    'mango': 'mango',
    'pomegranate': 'granada',
    'rhubarb': 'ruibarbo',

    # Maderas y resinas
    'sandalwood': 'sándalo',
    'cedar': 'cedro',
    'virginian cedar': 'cedro de virginia',
    'guaiac wood': 'madera de guayaco',
    'vetiver': 'vetiver',
    'patchouli': 'pachulí',
    'oud': 'oud',
    'oakmoss': 'musgo de roble',
    'cashmeran': 'cashmerán',
    'elemi': 'elemi',
    'benzoin': 'benjuí',
    'labdanum': 'ládano',
    'opoponax': 'opopónaco',
    'styrax': 'estoraque',

    # Especias
    'pepper': 'pimienta',
    'pink pepper': 'pimienta rosa',
    'cardamom': 'cardamomo',
    'cinnamon': 'canela',
    'clove': 'clavo',
    'nutmeg': 'nuez moscada',
    'ginger': 'jengibre',
    'saffron': 'azafrán',

    # Dulces
    'vanilla': 'vainilla',
    'tonka bean': 'haba tonka',
    'caramel': 'caramelo',
    'chocolate': 'chocolate',
    'coffee': 'café',
    'praline': 'praliné',
    'honey': 'miel',
    'almond': 'almendra',
    'coconut': 'coco',
    'toffee': 'toffee',
    'marshmallow': 'malvavisco',
    'condensed milk': 'leche condensada',

    # Otros
    'incense': 'incienso',
    'tobacco': 'tabaco',
    'suede': 'ante',
    'salt': 'sal',
    'ink': 'tinta',
    'ozone': 'ozono',
    'smoky': 'ahumado',
    'earthy': 'terroso',
    'balsamic': 'balsámico',
    'animalic': 'animal'
}


TRADUCCION_ACORDES = {

    'woody': 'madera',
    'amber': 'ámbar',
    'aromatic': 'aromático',
    'citrus': 'cítrico',
    'floral': 'floral',
    'fresh': 'fresco',
    'green': 'verde',
    'fruity': 'frutal',
    'gourmand': 'gourmand',
    'leather': 'cuero',
    'chypre': 'chypre',
    'oriental': 'oriental',
    'powdery': 'polvoso',
    'musky': 'almizclado',
    'aquatic': 'acuático',
    'herbal': 'herbal',

    'sweet': 'dulce',
    'warm spicy': 'especiado cálido',
    'soft spicy': 'especiado suave',
    'fresh spicy': 'especiado fresco',
    'lactonic': 'lactónico',
    'creamy': 'cremoso',
    'soapy': 'jabonoso',
    'metallic': 'metálico',
    'mineral': 'mineral',
    'ozonic': 'ozónico',
    'smoky': 'ahumado',
    'earthy': 'terroso',
    'balsamic': 'balsámico',
    'animalic': 'animal',
    'tobacco': 'tabaco',
    'nutty': 'avellanado',
    'cacao': 'cacao'
}


TRADUCCION_GENEROS = {
    'for-men': 'para hombre',
    'for-women': 'para mujer',
    'unisex': 'unisex',
    'for-her': 'para mujer',
    'for-him': 'para hombre',
    'masculine': 'masculino',
    'feminine': 'femenino',
    'men': 'hombre',
    'women': 'mujer'
}

# --------------------------------------------------
# Diccionarios de traducción inversa (castellano -> inglés)
# --------------------------------------------------
TRADUCCION_INVERSA_NOTAS = {v: k for k, v in TRADUCCION_NOTAS.items()}
TRADUCCION_INVERSA_ACORDES = {v: k for k, v in TRADUCCION_ACORDES.items()}
TRADUCCION_INVERSA_GENEROS = {v: k for k, v in TRADUCCION_GENEROS.items()}


# --------------------------------------------------
# Funciones de traducción
# --------------------------------------------------
def traducir_texto(texto, diccionario):
    """Traduce un texto usando el diccionario proporcionado"""
    if not texto or not isinstance(texto, str):
        return texto

    texto_lower = texto.lower().strip()

    # Buscar traducción exacta
    if texto_lower in diccionario:
        return diccionario[texto_lower]

    # Buscar traducción parcial (para textos compuestos)
    palabras = texto_lower.split()
    traducido = []
    for palabra in palabras:
        if palabra in diccionario:
            traducido.append(diccionario[palabra])
        else:
            traducido.append(palabra)

    return ' '.join(traducido)


def traducir_texto_inverso(texto, diccionario):
    """Traduce de castellano a inglés usando el diccionario inverso"""
    if not texto or not isinstance(texto, str):
        return texto

    texto_lower = texto.lower().strip()

    # Buscar traducción exacta
    if texto_lower in diccionario:
        return diccionario[texto_lower]

    # Buscar traducción parcial (para textos compuestos)
    palabras = texto_lower.split()
    traducido = []
    for palabra in palabras:
        if palabra in diccionario:
            traducido.append(diccionario[palabra])
        else:
            traducido.append(palabra)

    return ' '.join(traducido)


def traducir_lista(lista, diccionario):
    """Traduce una lista de textos"""
    if not isinstance(lista, list):
        return lista

    return [traducir_texto(str(item), diccionario) for item in lista if item]


def traducir_lista_inversa(lista, diccionario):
    """Traduce una lista de textos de castellano a inglés"""
    if not isinstance(lista, list):
        return lista

    return [traducir_texto_inverso(str(item), diccionario) for item in lista if item]


def traducir_perfume(perfume):
    """Aplica traducciones a todos los campos relevantes de un perfume"""
    perfume_traducido = perfume.copy()

    # Traducir género
    if 'genero' in perfume_traducido and perfume_traducido['genero']:
        perfume_traducido['genero'] = traducir_texto(
            str(perfume_traducido['genero']),
            TRADUCCION_GENEROS
        )

    # Traducir notas (salida, corazón, base)
    for campo_nota in ['salida', 'corazon', 'base']:
        if campo_nota in perfume_traducido and perfume_traducido[campo_nota]:
            # Si es una lista, traducir cada elemento
            if isinstance(perfume_traducido[campo_nota], list):
                perfume_traducido[campo_nota] = traducir_lista(
                    perfume_traducido[campo_nota],
                    TRADUCCION_NOTAS
                )
            # Si es string, traducir el texto completo
            elif isinstance(perfume_traducido[campo_nota], str):
                # Para strings, primero separar por comas y luego traducir
                notas_lista = [n.strip() for n in str(perfume_traducido[campo_nota]).split(',')]
                notas_traducidas = traducir_lista(notas_lista, TRADUCCION_NOTAS)
                perfume_traducido[campo_nota] = ', '.join(notas_traducidas)

    # Traducir acordes principales
    if 'main_accords' in perfume_traducido and perfume_traducido['main_accords']:
        perfume_traducido['main_accords'] = traducir_lista(
            perfume_traducido['main_accords'],
            TRADUCCION_ACORDES
        )

    return perfume_traducido


# --------------------------------------------------
# Función para preparar respuesta final (MODIFICADA)
# --------------------------------------------------
def preparar_respuesta(obj):
    """Aplica limpiezas, reemplazo de guiones y traducciones"""
    obj_limpio = limpiar_para_json(obj)
    obj_con_espacios = reemplazar_guiones_respuesta(obj_limpio)

    # Si es un perfume individual, traducirlo
    if isinstance(obj_con_espacios, dict):
        return traducir_perfume(obj_con_espacios)
    # Si es una lista de perfumes, traducir cada uno
    elif isinstance(obj_con_espacios, list) and obj_con_espacios and isinstance(obj_con_espacios[0], dict):
        return [traducir_perfume(perfume) for perfume in obj_con_espacios]

    return obj_con_espacios


# --------------------------------------------------
# Función para traducir parámetros de búsqueda
# --------------------------------------------------
def traducir_parametros_busqueda(notas_param=None, acordes_param=None, genero_param=None):
    """Traduce los parámetros de búsqueda de castellano a inglés"""
    parametros_traducidos = {}

    # Traducir notas
    if notas_param:
        notas_castellano = [n.strip().lower() for n in notas_param.split(",") if n.strip()]
        notas_ingles = []
        for nota in notas_castellano:
            # Primero intentar traducción inversa directa
            if nota in TRADUCCION_INVERSA_NOTAS:
                notas_ingles.append(TRADUCCION_INVERSA_NOTAS[nota])
            else:
                # Si no encuentra, buscar en los valores del diccionario
                encontrado = False
                for key_eng, value_esp in TRADUCCION_NOTAS.items():
                    if value_esp.lower() == nota:
                        notas_ingles.append(key_eng)
                        encontrado = True
                        break
                # Si no se encuentra, usar el original (por si ya está en inglés)
                if not encontrado:
                    notas_ingles.append(nota)

        parametros_traducidos['notas'] = notas_ingles
        print(f"🔍 Notas traducidas: {notas_castellano} -> {notas_ingles}")

    # Traducir acordes
    if acordes_param:
        acordes_castellano = [a.strip().lower() for a in acordes_param.split(",") if a.strip()]
        acordes_ingles = []
        for acorde in acordes_castellano:
            # Primero intentar traducción inversa directa
            if acorde in TRADUCCION_INVERSA_ACORDES:
                acordes_ingles.append(TRADUCCION_INVERSA_ACORDES[acorde])
            else:
                # Si no encuentra, buscar en los valores del diccionario
                encontrado = False
                for key_eng, value_esp in TRADUCCION_ACORDES.items():
                    if value_esp.lower() == acorde:
                        acordes_ingles.append(key_eng)
                        encontrado = True
                        break
                # Si no se encuentra, usar el original (por si ya está en inglés)
                if not encontrado:
                    acordes_ingles.append(acorde)

        parametros_traducidos['acordes'] = acordes_ingles
        print(f"🔍 Acordes traducidos: {acordes_castellano} -> {acordes_ingles}")

    # Traducir género
    if genero_param:
        genero_castellano = genero_param.lower().strip()
        if genero_castellano in TRADUCCION_INVERSA_GENEROS:
            parametros_traducidos['genero'] = TRADUCCION_INVERSA_GENEROS[genero_castellano]
            print(f"🔍 Género traducido: {genero_castellano} -> {parametros_traducidos['genero']}")
        else:
            # Buscar en los valores
            for key_eng, value_esp in TRADUCCION_GENEROS.items():
                if value_esp.lower() == genero_castellano:
                    parametros_traducidos['genero'] = key_eng
                    print(f"🔍 Género traducido: {genero_castellano} -> {key_eng}")
                    break
            # Si no se encuentra, usar el original
            if 'genero' not in parametros_traducidos:
                parametros_traducidos['genero'] = genero_param

    return parametros_traducidos


# --------------------------------------------------
# Función para cargar el CSV (MODIFICADA)
# --------------------------------------------------
def cargar_csv():
    archivo = obtener_ruta_csv()
    print(f"📁 Intentando cargar CSV desde: {archivo}")
    
    if not os.path.exists(archivo):
        print(f"❌ Error: {archivo} no encontrado.")
        print(f"📂 Directorio actual: {os.getcwd()}")
        print(f"📂 Archivos en directorio actual: {os.listdir('.')}")
        raise FileNotFoundError(f"El archivo {archivo} no se encuentra")

    configuraciones = [
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "latin1", "quotechar": '"'},
        {"sep": "\t", "encoding": "latin1"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"}
    ]

    for i, config in enumerate(configuraciones):
        try:
            print(f"🔧 Intentando configuración {i + 1}: {config}")
            df = pd.read_csv(archivo, **config)
            # Limpiar NaN/None del DataFrame inmediatamente
            df = df.fillna("")
            print(f"✅ CSV cargado exitosamente con configuración {i + 1}")
            print(f"📊 Dimensiones del DataFrame: {df.shape}")
            print(f"📋 Columnas: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Intento {i + 1} fallido: {e}")
            continue

    raise Exception("No se pudo cargar el CSV con ninguna configuración probada")


# --------------------------------------------------
# Cargar el DataFrame (SIN modificar guiones)
# --------------------------------------------------
try:
    df = cargar_csv()
    print("🎉 CSV cargado exitosamente")
    print(f"📊 Dimensiones del DataFrame: {df.shape}")
except Exception as e:
    print(f"❌ Error crítico al cargar el CSV: {e}")
    df = pd.DataFrame(columns=['url', 'perfume', 'marca', 'genero', 'año', 'salida', 'corazon', 'base', 'main_accords'])

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
print("📋 Columnas disponibles en la API:", CAMPOS_DISPONIBLES)


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


# Solo ejecutar esto si el df tiene datos
if not df.empty:
    df['todas_notas'] = df.apply(extraer_notas, axis=1)
    VOCAB = sorted({n for notas in df['todas_notas'] for n in notas})


    def vectorizar_notas(notas, vocab):
        return [1 if n in notas else 0 for n in vocab]


    MATRIZ_VECTORES = np.array([vectorizar_notas(notas, VOCAB) for notas in df['todas_notas']])
else:
    MATRIZ_VECTORES = np.array([])


# --------------------------------------------------
# Endpoints MODIFICADOS - Ahora con traducción de parámetros de búsqueda
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

        # Aplicar reemplazo de guiones por espacios y traducciones
        perfumes_limpios = [preparar_respuesta(perfume) for perfume in perfumes]

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
    perfume_limpio = preparar_respuesta(perfume)
    return jsonify(perfume_limpio)

@app.route('/perfumes/top-rated', methods=['POST'])
def get_top_rated():
    try:
        data = request.json or {}
        limit = data.get('limit', 100)
        min_votes = data.get('min_votes', 1)
        sort_by = data.get('sort_by', 'average_rating')
        order = data.get('order', 'desc')
        
     
        return jsonify([])  # Retornar array de perfumes
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/perfumes/all', methods=['GET'])
def get_all_perfumes():
    try:
        # Retornar todos los perfumes
        return jsonify([])  # Tu lista de perfumes
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

        # --- TRADUCIR PARÁMETROS DE BÚSQUEDA ---
        notas_param = request.args.get('nota')
        acordes_param = request.args.get('acorde')
        genero_param = request.args.get('genero')

        parametros_traducidos = traducir_parametros_busqueda(notas_param, acordes_param, genero_param)

        # --- Filtros básicos CONVIRTIENDO ESPACIOS A GUIONES PARA BÚSQUEDA ---
        for param, columna in filtros_texto.items():
            valor = request.args.get(param)
            if valor and columna in query.columns:
                # Para marcas y perfumes, convertir espacios a guiones para buscar en la base de datos
                if columna in ['marca', 'perfume']:
                    # Crear versión con guiones (como está en la base de datos)
                    valor_con_guiones = valor.replace(" ", "-")
                    print(f"🔍 Buscando {columna}: '{valor}' -> '{valor_con_guiones}'")
                    # Buscar por la versión con guiones usando contains para nombres de perfume
                    if columna == 'marca':
                        query = query[query[columna].astype(str).str.lower() == valor_con_guiones.lower()]
                    else:  # Para perfume, usar contains para búsqueda parcial
                        query = query[query[columna].astype(str).str.contains(valor_con_guiones, case=False, na=False)]
                elif columna == 'genero':
                    # Usar el género traducido si está disponible
                    genero_buscar = parametros_traducidos.get('genero', valor)
                    query = query[query[columna].astype(str).str.lower() == genero_buscar.lower()]
                else:
                    query = query[query[columna].astype(str).str.contains(valor, case=False, na=False)]

        # --- Buscar por notas (USANDO TÉRMINOS TRADUCIDOS) ---
        if 'notas' in parametros_traducidos:
            notas_buscar = parametros_traducidos['notas']
            # Convertir espacios a guiones para coincidir con la base de datos
            notas_buscar = [n.strip().lower().replace(" ", "-") for n in notas_buscar]
            print(f"🔍 Buscando notas (traducidas): {notas_buscar}")

            def contiene_todas(row):
                notas_perfume = extraer_notas(row)
                return all(n in notas_perfume for n in notas_buscar)

            query = query[query.apply(contiene_todas, axis=1)]

        # --- Buscar por acordes (USANDO TÉRMINOS TRADUCIDOS) ---
        if 'acordes' in parametros_traducidos and 'main_accords' in query.columns:
            acordes_buscar = parametros_traducidos['acordes']
            # Convertir espacios a guiones para coincidir con la base de datos
            acordes_buscar = [a.strip().lower().replace(" ", "-") for a in acordes_buscar]
            print(f"🔍 Buscando acordes (traducidas): {acordes_buscar}")

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
        resultados_limpios = [preparar_respuesta(resultado) for resultado in resultados]

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
            'parametros_traducidos': parametros_traducidos,
            'resultados': resultados_limpios
        })

    except Exception as e:
        abort(500, description=f"Error interno en la búsqueda: {str(e)}")


@app.route('/perfumes/similares', methods=['GET'])
def get_similares_nombre():
    nombre = request.args.get('nombre')
    if not nombre:
        abort(400, description="Debes proporcionar el parámetro 'nombre'")
    
    # Obtener umbral de similitud (porcentaje, default 70%)
    umbral_param = request.args.get('umbral', 70)
    try:
        umbral = float(umbral_param) / 100.0  # Convertir porcentaje a decimal (70% -> 0.7)
    except ValueError:
        umbral = 0.7  # Valor por defecto si no es válido

    # CONVERTIR ESPACIOS A GUIONES PARA BÚSQUEDA
    nombre_con_guiones = nombre.replace(" ", "-")
    print(f"🔍 Buscando similares para: '{nombre}' -> '{nombre_con_guiones}'")
    print(f"🔍 Umbral de similitud: {umbral_param}% ({umbral})")

    coincidencias = df[df['perfume'].astype(str).str.contains(nombre_con_guiones, case=False, na=False)]
    if coincidencias.empty:
        abort(404, description=f"No se encontró ningún perfume que coincida con '{nombre}'")

    idx_base = coincidencias.index[0]
    base_vec = MATRIZ_VECTORES[idx_base].reshape(1, -1)

    similitudes = cosine_similarity(base_vec, MATRIZ_VECTORES)[0]
    df['score_similaridad'] = similitudes

    # FILTRAR por umbral y excluir el perfume base
    similares = df[(df.index != idx_base) & (df['score_similaridad'] >= umbral)]
    
    # Ordenar por similitud
    similares = similares.sort_values('score_similaridad', ascending=False)
    
    # Limitar cantidad si se especifica
    top_n = int(request.args.get('n', 50))  # Aumentar default para que el filtro funcione
    similares = similares.head(top_n)

    similares_out = filtrar_campos(similares).copy()
    similares_out['similitud'] = (similares['score_similaridad'] * 100).round(1).astype(str)

    base_limpio = preparar_respuesta(filtrar_campos(df.iloc[[idx_base]]).iloc[0].to_dict())
    similares_limpios = [preparar_respuesta(perfume) for perfume in similares_out.to_dict(orient='records')]

    print(f"✅ Encontrados {len(similares_limpios)} perfumes con similitud >= {umbral_param}%")

    return jsonify({
        'termino_buscado': nombre,
        'base': base_limpio,
        'umbral': umbral_param,
        'similares': similares_limpios
    })


# --------------------------------------------------
# Endpoint: Obtener todas las marcas únicas
# --------------------------------------------------
@app.route('/perfumes/marcas', methods=['GET'])
def get_marcas():
    try:
        # Obtener todas las marcas únicas, ordenadas alfabéticamente
        marcas = df['marca'].dropna().unique()
        marcas = [marca for marca in marcas if str(marca).strip() != '']

        # Aplicar reemplazo de guiones por espacios en la respuesta
        marcas_con_espacios = [marca.replace("-", " ") if isinstance(marca, str) else marca for marca in marcas]
        marcas_ordenadas = sorted(marcas_con_espacios, key=lambda x: str(x).lower())

        return jsonify({
            'total_marcas': len(marcas_ordenadas),
            'marcas': marcas_ordenadas
        })
    except Exception as e:
        abort(500, description=f"Error al obtener las marcas: {str(e)}")


# --------------------------------------------------
# Endpoint: Obtener perfumes por marca
# --------------------------------------------------
@app.route('/perfumes/marca/<string:nombre_marca>', methods=['GET'])
def get_perfumes_por_marca(nombre_marca):
    try:
        # CONVERTIR ESPACIOS A GUIONES PARA BÚSQUEDA
        nombre_marca_con_guiones = nombre_marca.replace(" ", "-")
        print(f"🔍 Buscando perfumes por marca: '{nombre_marca}' -> '{nombre_marca_con_guiones}'")

        pagina = int(request.args.get('pagina', 1))
        por_pagina = int(request.args.get('por_pagina', 20))

        if pagina < 1 or por_pagina < 1:
            abort(400, description="Los parámetros de paginación deben ser positivos")

        # Filtrar por marca usando la versión con guiones (como está en la base de datos)
        perfumes_marca = df[df['marca'].astype(str).str.lower() == nombre_marca_con_guiones.lower()]

        if perfumes_marca.empty:
            print(f"⚠️ No se encontraron perfumes para la marca: '{nombre_marca_con_guiones}'")
            return jsonify({
                'marca_solicitada': nombre_marca,
                'marca_buscada': nombre_marca_con_guiones,
                'total_perfumes': 0,
                'pagina': pagina,
                'por_pagina': por_pagina,
                'perfumes': []
            })

        # Paginación
        inicio = (pagina - 1) * por_pagina
        fin = inicio + por_pagina
        subset = perfumes_marca.iloc[inicio:fin]

        perfumes = filtrar_campos(subset).to_dict(orient='records')
        perfumes_limpios = [preparar_respuesta(perfume) for perfume in perfumes]

        print(f"✅ Encontrados {len(perfumes_marca)} perfumes para '{nombre_marca_con_guiones}'")

        return jsonify({
            'marca_solicitada': nombre_marca,
            'marca_encontrada': nombre_marca_con_guiones,
            'total_perfumes': len(perfumes_marca),
            'pagina': pagina,
            'por_pagina': por_pagina,
            'perfumes': perfumes_limpios
        })
    except ValueError:
        abort(400, description="Parámetros de paginación inválidos")
    except Exception as e:
        abort(500, description=f"Error al obtener perfumes de la marca: {str(e)}")


# --------------------------------------------------
# Endpoint para obtener traducciones disponibles
# --------------------------------------------------
@app.route('/perfumes/traducciones', methods=['GET'])
def get_traducciones():
    """Endpoint para obtener todos los diccionarios de traducción"""
    return jsonify({
        'generos': TRADUCCION_GENEROS,
        'notas': TRADUCCION_NOTAS,
        'acordes': TRADUCCION_ACORDES,
        'traducciones_inversas': {
            'notas': TRADUCCION_INVERSA_NOTAS,
            'acordes': TRADUCCION_INVERSA_ACORDES,
            'generos': TRADUCCION_INVERSA_GENEROS
        }
    })


# --------------------------------------------------
# Endpoint de diagnóstico para ver marcas disponibles
# --------------------------------------------------
@app.route('/perfumes/debug/marcas', methods=['GET'])
def debug_marcas():
    """Endpoint para diagnosticar qué marcas están disponibles"""
    try:
        # Obtener todas las marcas únicas
        marcas = df['marca'].dropna().unique()
        marcas = [marca for marca in marcas if str(marca).strip() != '']

        # Separar marcas con guiones y sin guiones
        marcas_con_guiones = [m for m in marcas if '-' in str(m)]
        marcas_sin_guiones = [m for m in marcas if '-' not in str(m)]

        # Crear mapeo de marcas con guiones a sus versiones con espacios
        mapeo_marcas = {marca: marca.replace("-", " ") for marca in marcas_con_guiones}

        return jsonify({
            'total_marcas': len(marcas),
            'marcas_con_guiones': marcas_con_guiones,
            'marcas_sin_guiones': marcas_sin_guiones,
            'mapeo_guiones_a_espacios': mapeo_marcas,
            'ejemplos_busqueda': [
                {
                    'marca_con_espacios': 'Yves Saint Laurent',
                    'marca_con_guiones': 'Yves-Saint-Laurent',
                    'url_ejemplo': '/perfumes/marca/Yves Saint Laurent'
                },
                {
                    'marca_con_espacios': 'Jean Paul Gaultier',
                    'marca_con_guiones': 'Jean-Paul-Gaultier',
                    'url_ejemplo': '/perfumes/marca/Jean Paul Gaultier'
                }
            ]
        })
    except Exception as e:
        abort(500, description=f"Error en diagnóstico: {str(e)}")


# --------------------------------------------------
# Endpoint de diagnóstico para ver notas y acordes disponibles
# --------------------------------------------------
@app.route('/perfumes/debug/notas-acordes', methods=['GET'])
def debug_notas_acordes():
    """Endpoint para diagnosticar qué notas y acordes están disponibles"""
    try:
        # Obtener todas las notas únicas
        todas_notas = []
        for notas in df['todas_notas']:
            todas_notas.extend(notas)

        notas_unicas = sorted(list(set(todas_notas)))

        # Separar notas con guiones y sin guiones
        notas_con_guiones = [n for n in notas_unicas if '-' in str(n)]
        notas_sin_guiones = [n for n in notas_unicas if '-' not in str(n)]

        # Crear mapeo de notas con guiones a sus versiones con espacios
        mapeo_notas = {nota: notas.replace("-", " ") for notas in notas_con_guiones}

        # Obtener acordes únicos
        todos_acordes = []
        for acordes in df['main_accords']:
            if isinstance(acordes, list):
                todos_acordes.extend([str(a) for a in acordes])

        acordes_unicos = sorted(list(set(todos_acordes)))

        # Separar acordes con guiones y sin guiones
        acordes_con_guiones = [a for a in acordes_unicos if '-' in str(a)]
        acordes_sin_guiones = [a for a in acordes_unicos if '-' not in str(a)]

        # Crear mapeo de acordes con guiones a sus versiones con espacios
        mapeo_acordes = {acorde: acorde.replace("-", " ") for acorde in acordes_con_guiones}

        return jsonify({
            'total_notas': len(notas_unicas),
            'notas_con_guiones': notas_con_guiones,
            'notas_sin_guiones': notas_sin_guiones,
            'mapeo_notas_guiones_a_espacios': mapeo_notas,
            'total_acordes': len(acordes_unicos),
            'acordes_con_guiones': acordes_con_guiones,
            'acordes_sin_guiones': acordes_sin_guiones,
            'mapeo_acordes_guiones_a_espacios': mapeo_acordes
        })
    except Exception as e:
        abort(500, description=f"Error en diagnóstico: {str(e)}")


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
# Endpoint de información del sistema
# --------------------------------------------------
@app.route('/system/info', methods=['GET'])
def system_info():
    """Endpoint para obtener información del sistema y rutas"""
    return jsonify({
        'directorio_actual': os.getcwd(),
        'directorio_script': os.path.dirname(os.path.abspath(__file__)),
        'ruta_csv_intentada': obtener_ruta_csv(),
        'existe_csv': os.path.exists(obtener_ruta_csv()),
        'archivos_en_directorio': os.listdir('.'),
        'dimensiones_dataframe': df.shape if not df.empty else 'Vacío',
        'columnas_dataframe': list(df.columns) if not df.empty else []
    })


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask...")
    print(f"📂 Directorio actual: {os.getcwd()}")
    print(f"📂 Directorio del script: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"📁 Ruta del CSV: {obtener_ruta_csv()}")
    print(f"✅ CSV existe: {os.path.exists(obtener_ruta_csv())}")
    
    if not df.empty:
        print(f"📊 DataFrame cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    else:
        print("⚠️ DataFrame vacío - funcionando en modo limitado")
    
    app.run(debug=True, host='0.0.0.0', port=5000)