const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const db = require('./db');

const app = express();

// Configuración de CORS para permitir peticiones desde React
app.use(cors());
app.use(express.json());

// ==========================================
//  HELPER: MANEJO DE ESPACIOS Y GUIONES
// ==========================================
// Esta función crea variaciones para que "Tom Ford" encuentre "Tom-Ford" y viceversa.
const getVariations = (text) => {
  if (!text) return [];
  const decoded = decodeURIComponent(text);
  const withSpaces = decoded.replace(/-/g, ' ');
  const withHyphens = decoded.replace(/ /g, '-');
  // Devolvemos ambas versiones para asegurar que lo encuentre en BD
  return [withSpaces, withHyphens];
};

// ==========================================
//  RUTAS DE USUARIOS (AUTH)
// ==========================================

// --- REGISTRO ---
app.post('/register', async (req, res) => {
  const { username, password, name, lastname } = req.body;
  
  if (!username || !password || !name) {
    return res.status(400).json({ error: 'Faltan datos obligatorios' });
  }

  try {
    // Verificar si existe
    const [existing] = await db.query('SELECT * FROM usuario WHERE username = ?', [username]);
    if (existing.length > 0) return res.status(400).json({ error: 'El usuario ya existe' });

    // Encriptar contraseña
    const salt = await bcrypt.genSalt(10);
    const hash = await bcrypt.hash(password, salt);

    // Guardar
    await db.query(
      'INSERT INTO usuario (username, password_hash, nombre, apellido) VALUES (?, ?, ?, ?)',
      [username, hash, name, lastname]
    );
    
    res.json({ message: 'Usuario registrado correctamente' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al registrar en base de datos' });
  }
});

// --- LOGIN ---
app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  
  if (!username || !password) {
    return res.status(400).json({ error: 'Faltan credenciales' });
  }

  try {
    const [users] = await db.query('SELECT * FROM usuario WHERE username = ?', [username]);
    
    if (users.length === 0) return res.status(400).json({ error: 'Usuario no encontrado' });

    const user = users[0];
    const validPassword = await bcrypt.compare(password, user.password_hash);

    if (!validPassword) return res.status(400).json({ error: 'Contraseña incorrecta' });

    // Mapeo de datos para React
    res.json({
      id: user.id,
      username: user.username,
      name: user.nombre,
      lastname: user.apellido
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error del servidor' });
  }
});

// ==========================================
//  RUTAS DE COMENTARIOS
// ==========================================

// --- OBTENER COMENTARIOS ---
app.get('/comentarios/:perfumeId', async (req, res) => {
  const { perfumeId } = req.params;
  try {
    const query = `
      SELECT c.*, u.username, u.nombre 
      FROM comentario c
      JOIN usuario u ON c.usuario_id = u.id
      WHERE c.perfume_id = ?
      ORDER BY c.fecha_creacion DESC
    `;
    const [comments] = await db.query(query, [perfumeId]);
    res.json(comments);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al cargar comentarios' });
  }
});

// --- GUARDAR COMENTARIO ---
app.post('/comentarios', async (req, res) => {
  const { usuario_id, perfume_id, texto, puntuacion } = req.body;

  if (!usuario_id || !perfume_id || !texto) {
    return res.status(400).json({ error: 'Faltan datos' });
  }

  try {
    await db.query(
      'INSERT INTO comentario (usuario_id, perfume_id, texto, puntuacion) VALUES (?, ?, ?, ?)',
      [usuario_id, perfume_id, texto, puntuacion]
    );
    res.json({ message: 'Comentario guardado' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al guardar comentario' });
  }
});

// ==========================================
//  RUTAS DE DATOS (PERFUMES DESDE MYSQL)
// ==========================================

// --- 1. OBTENER MARCAS ---
app.get('/perfumes/marcas', async (req, res) => {
  try {
    const [marcas] = await db.query('SELECT nombre FROM marca ORDER BY nombre ASC');
    const listaMarcas = marcas.map(m => m.nombre);
    res.json({ marcas: listaMarcas });
  } catch (err) {
    res.status(500).json({ error: 'Error obteniendo marcas' });
  }
});

// --- 2. PERFUMES POR MARCA (ROBUSTO) ---
app.get('/perfumes/marca/:nombreMarca', async (req, res) => {
  const { nombreMarca } = req.params;
  
  // Obtenemos variaciones (con espacio y con guion) para asegurar la búsqueda
  const vars = getVariations(nombreMarca);
  
  console.log(`🔍 Buscando marca: "${vars[0]}" o "${vars[1]}"`);

  try {
    const query = `
      SELECT p.id, p.nombre as perfume, m.nombre as marca, g.nombre as genero, 
             p.año, p.notas_salida as salida, p.notas_corazon as corazon, 
             p.notas_base as base, p.acordes_principales as main_accords,
             pe.nombre as perfumista
      FROM perfume p
      JOIN marca m ON p.marca_id = m.id
      LEFT JOIN genero g ON p.genero_id = g.id
      LEFT JOIN perfumista pe ON p.perfumista_id = pe.id
      WHERE m.nombre LIKE ? OR m.nombre LIKE ?
    `;
    
    const [perfumes] = await db.query(query, [`${vars[0]}`, `${vars[1]}`]);
    res.json({ perfumes });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error obteniendo perfumes' });
  }
});

// --- 3. DETALLE DE PERFUME ---
app.get('/perfumes/:id', async (req, res) => {
  try {
    const query = `
      SELECT p.id, p.nombre as perfume, m.nombre as marca, g.nombre as genero, 
             p.año, p.notas_salida as salida, p.notas_corazon as corazon, 
             p.notas_base as base, p.acordes_principales as main_accords,
             pe.nombre as perfumista
      FROM perfume p
      JOIN marca m ON p.marca_id = m.id
      LEFT JOIN genero g ON p.genero_id = g.id
      LEFT JOIN perfumista pe ON p.perfumista_id = pe.id
      WHERE p.id = ?
    `;
    const [rows] = await db.query(query, [req.params.id]);
    if (rows.length === 0) return res.status(404).json({ error: 'No encontrado' });
    res.json(rows[0]);
  } catch (err) {
    res.status(500).json({ error: 'Error servidor' });
  }
});

// --- 4. BÚSQUEDA AVANZADA (ROBUSTO) ---
app.get('/perfumes/search', async (req, res) => {
  const { perfume, marca, genero, nota, acorde, perfumista } = req.query; 
  
  try {
    let sql = `
      SELECT p.id, p.nombre as perfume, m.nombre as marca, g.nombre as genero, p.año,
             p.notas_salida, p.notas_corazon, p.notas_base, p.acordes_principales
      FROM perfume p
      JOIN marca m ON p.marca_id = m.id
      LEFT JOIN genero g ON p.genero_id = g.id
      LEFT JOIN perfumista pe ON p.perfumista_id = pe.id
      WHERE 1=1
    `;
    const params = [];

    // Lógica flexible para cada campo
    if (perfume) {
      const vars = getVariations(perfume);
      sql += " AND (p.nombre LIKE ? OR p.nombre LIKE ?)";
      params.push(`%${vars[0]}%`, `%${vars[1]}%`);
    }
    if (marca) {
      const vars = getVariations(marca);
      sql += " AND (m.nombre LIKE ? OR m.nombre LIKE ?)";
      params.push(`%${vars[0]}%`, `%${vars[1]}%`);
    }
    if (genero) {
      const vars = getVariations(genero);
      sql += " AND (g.nombre LIKE ? OR g.nombre LIKE ?)";
      params.push(`%${vars[0]}%`, `%${vars[1]}%`);
    }
    if (perfumista) {
      const vars = getVariations(perfumista);
      sql += " AND (pe.nombre LIKE ? OR pe.nombre LIKE ?)";
      params.push(`%${vars[0]}%`, `%${vars[1]}%`);
    }
    // Notas en cualquiera de las 3 columnas
    if (nota) {
      const vars = getVariations(nota);
      sql += ` AND (
        p.notas_salida LIKE ? OR p.notas_salida LIKE ? OR
        p.notas_corazon LIKE ? OR p.notas_corazon LIKE ? OR
        p.notas_base LIKE ? OR p.notas_base LIKE ?
      )`;
      params.push(`%${vars[0]}%`, `%${vars[1]}%`, `%${vars[0]}%`, `%${vars[1]}%`, `%${vars[0]}%`, `%${vars[1]}%`);
    }
    if (acorde) {
      const vars = getVariations(acorde);
      sql += " AND (p.acordes_principales LIKE ? OR p.acordes_principales LIKE ?)";
      params.push(`%${vars[0]}%`, `%${vars[1]}%`);
    }
    
    sql += " LIMIT 50"; // Limitar resultados

    const [resultados] = await db.query(sql, params);
    res.json({ resultados });

  } catch (err) {
    console.error("Error en búsqueda:", err);
    res.status(500).json({ error: 'Error realizando la búsqueda' });
  }
});

// --- 5. CLONES / SIMILARES (SQL SIMPLE) ---
app.get('/perfumes/similares', async (req, res) => {
  const { nombre } = req.query;
  if (!nombre) return res.status(400).json({ error: 'Falta nombre' });

  const vars = getVariations(nombre);

  try {
    // 1. Encontrar el original
    const queryId = "SELECT id, marca_id, acordes_principales FROM perfume WHERE nombre LIKE ? OR nombre LIKE ? LIMIT 1";
    const [originals] = await db.query(queryId, [`%${vars[0]}%`, `%${vars[1]}%`]);

    if (originals.length === 0) return res.status(404).json({ similares: [] });
    const original = originals[0];

    // 2. Buscar similares por marca o acorde (Aproximación SQL)
    const primerAcorde = original.acordes_principales ? original.acordes_principales.split(',')[0].trim() : '';
    
    let sqlSimilares = `
      SELECT p.id, p.nombre as perfume, m.nombre as marca, '85%' as similitud
      FROM perfume p
      JOIN marca m ON p.marca_id = m.id
      WHERE p.id != ? 
      AND (p.marca_id = ? OR p.acordes_principales LIKE ?)
      LIMIT 10
    `;
    
    const [similares] = await db.query(sqlSimilares, [original.id, original.marca_id, `%${primerAcorde}%`]);
    res.json({ similares });

  } catch (err) {
    res.status(500).json({ error: 'Error buscando similares' });
  }
});

// ==========================================
//  ARRANQUE DEL SERVIDOR
// ==========================================
// Usamos el puerto 3001 para no chocar con el 5000 (por si acaso usas Python)
const PORT = 3001;
app.listen(PORT, () => {
  console.log(`🚀 Servidor backend completo corriendo en http://localhost:${PORT}`);
});