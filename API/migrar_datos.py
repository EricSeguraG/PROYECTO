import pandas as pd
import mysql.connector
import os

# --- ⚠️ CONFIGURACIÓN: PON TU CONTRASEÑA DE MYSQL AQUÍ ⚠️ ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root', 
    'database': 'essence'
}

CSV_FILE = 'fra_perfumes.csv' 

def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

def clean_data(val):
    """Limpia datos vacíos (NaN) del CSV"""
    if pd.isna(val) or val == "" or str(val).lower() == "nan": 
        return None
    return str(val).strip()

def migrar():
    print("🚀 Iniciando migración de datos a MySQL...")
    
    # 1. Cargar CSV
    try:
        # Intentamos leer con diferentes codificaciones por si acaso
        try:
            df = pd.read_csv(CSV_FILE, encoding='latin1', sep=None, engine='python')
        except:
            df = pd.read_csv(CSV_FILE, encoding='utf-8', sep=None, engine='python')
            
        print(f"✅ CSV Cargado: {len(df)} perfumes encontrados.")
    except Exception as e:
        print(f"❌ Error leyendo el archivo {CSV_FILE}: {e}")
        print("Asegúrate de que el archivo esté en la misma carpeta que este script.")
        return

    try:
        conn = connect_db()
        cursor = conn.cursor()
        print("✅ Conectado a MySQL.")
    except Exception as e:
        print(f"❌ Error conectando a MySQL: {e}")
        return

    # 2. Migrar Marcas (Brands)
    print("📦 Migrando Marcas...")
    marcas = df['marca'].dropna().unique()
    for marca in marcas:
        if not marca: continue
        cursor.execute("INSERT IGNORE INTO marca (nombre) VALUES (%s)", (str(marca).strip(),))
    conn.commit()
    print("✅ Marcas guardadas.")

    # 3. Migrar Géneros
    print("👫 Migrando Géneros...")
    generos = df['genero'].dropna().unique()
    for gen in generos:
        if not gen: continue
        cursor.execute("INSERT IGNORE INTO genero (nombre) VALUES (%s)", (str(gen).strip(),))
    conn.commit()
    print("✅ Géneros guardados.")

    # 4. Migrar Perfumistas
    print("🧪 Migrando Perfumistas...")
    perfumistas = df['perfumista'].dropna().unique()
    for p in perfumistas:
        if not p: continue
        cursor.execute("INSERT IGNORE INTO perfumista (nombre) VALUES (%s)", (str(p).strip(),))
    conn.commit()
    print("✅ Perfumistas guardados.")

    # 5. Migrar Perfumes
    print("💎 Insertando Perfumes (esto tomará unos segundos)...")
    
    # Pre-cargar IDs para no hacer miles de consultas
    cursor.execute("SELECT nombre, id FROM marca")
    marca_map = {row[0].lower(): row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT nombre, id FROM genero")
    genero_map = {row[0].lower(): row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT nombre, id FROM perfumista")
    perfumista_map = {row[0].lower(): row[1] for row in cursor.fetchall()}

    count = 0
    errores = 0

    sql_perfume = """
        INSERT INTO perfume 
        (nombre, marca_id, genero_id, perfumista_id, año, notas_salida, notas_corazon, notas_base, acordes_principales)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        nombre = clean_data(row.get('perfume'))
        marca_nom = clean_data(row.get('marca'))
        
        if not nombre or not marca_nom: continue

        # Obtener IDs
        marca_id = marca_map.get(marca_nom.lower()) if marca_nom else None
        
        genero_nom = clean_data(row.get('genero'))
        genero_id = genero_map.get(genero_nom.lower()) if genero_nom else None
        
        perfumista_nom = clean_data(row.get('perfumista'))
        perfumista_id = perfumista_map.get(perfumista_nom.lower()) if perfumista_nom else None

        # Datos de texto
        anio = clean_data(row.get('año'))
        salida = clean_data(row.get('salida'))
        corazon = clean_data(row.get('corazon'))
        base = clean_data(row.get('base'))
        
        # Combinar main_accords si vienen en columnas separadas o una sola
        if 'main_accords' in df.columns:
            acordes = clean_data(row.get('main_accords'))
        else:
            # Si el CSV tiene columnas tipo 'MainAccord1', 'MainAccord2'... unirlas
            cols_acordes = [c for c in df.columns if 'accord' in c.lower()]
            vals_acordes = [str(row[c]) for c in cols_acordes if pd.notna(row[c])]
            acordes = ", ".join(vals_acordes)

        if marca_id: # Solo insertamos si existe la marca
            try:
                val = (nombre, marca_id, genero_id, perfumista_id, anio, salida, corazon, base, acordes)
                cursor.execute(sql_perfume, val)
                count += 1
            except Exception as err:
                # print(f"⚠️ Error en {nombre}: {err}") # Descomentar para ver errores específicos
                errores += 1

    conn.commit()
    print(f"🎉 ¡TERMINADO! Se insertaron {count} perfumes correctamente.")
    if errores > 0:
        print(f"⚠️ Hubo {errores} perfumes que no se pudieron insertar (posiblemente duplicados o faltaban datos).")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    migrar()