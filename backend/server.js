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
const getVariations = (text) => {
  if (!text) return [];
  const decoded = decodeURIComponent(text);
  const withSpaces = decoded.replace(/-/g, ' ');
  const withHyphens = decoded.replace(/ /g, '-');
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
//  RUTAS DE COMENTARIOS (ACTUALIZADAS)
// ==========================================

// --- OBTENER COMENTARIOS (MÁS ROBUSTA) ---
app.get('/comentarios/:perfumeIdentifier', async (req, res) => {
  const { perfumeIdentifier } = req.params;
  
  try {
    let query, params;
    
    // Si es un número, buscamos por ID de perfume
    if (!isNaN(perfumeIdentifier)) {
      query = `
        SELECT c.id, c.texto, c.puntuacion, c.fecha_creacion,
               u.id as usuario_id, u.username, u.nombre, u.apellido
        FROM comentario c
        JOIN usuario u ON c.usuario_id = u.id
        WHERE c.perfume_id = ?
        ORDER BY c.fecha_creacion DESC
      `;
      params = [perfumeIdentifier];
    } else {
      // Si es texto, buscamos por nombre del perfume
      const vars = getVariations(perfumeIdentifier);
      query = `
        SELECT c.id, c.texto, c.puntuacion, c.fecha_creacion,
               u.id as usuario_id, u.username, u.nombre, u.apellido
        FROM comentario c
        JOIN usuario u ON c.usuario_id = u.id
        JOIN perfume p ON c.perfume_id = p.id
        WHERE p.nombre LIKE ? OR p.nombre LIKE ?
        ORDER BY c.fecha_creacion DESC
      `;
      params = [`%${vars[0]}%`, `%${vars[1]}%`];
    }
    
    const [comments] = await db.query(query, params);
    res.json(comments);
  } catch (err) {
    console.error('Error cargando comentarios:', err);
    res.status(500).json({ error: 'Error al cargar comentarios' });
  }
});

// --- GUARDAR COMENTARIO (ACTUALIZADA) ---
app.post('/comentarios', async (req, res) => {
  const { usuario_id, perfume_id, perfume_name, texto, puntuacion } = req.body;

  if (!usuario_id || !texto) {
    return res.status(400).json({ error: 'Faltan datos obligatorios' });
  }

  try {
    let finalPerfumeId = perfume_id;
    
    // Si tenemos nombre pero no ID, buscamos el perfume
    if (perfume_name && !perfume_id) {
      const vars = getVariations(perfume_name);
      const [perfumes] = await db.query(
        'SELECT id FROM perfume WHERE nombre LIKE ? OR nombre LIKE ? LIMIT 1',
        [`%${vars[0]}%`, `%${vars[1]}%`]
      );
      
      if (perfumes.length > 0) {
        finalPerfumeId = perfumes[0].id;
      } else {
        // Si el perfume no existe en la BD, lo creamos temporalmente
        const [result] = await db.query(
          'INSERT INTO perfume (nombre) VALUES (?)',
          [perfume_name]
        );
        finalPerfumeId = result.insertId;
        console.log(`✅ Perfume temporal creado: ${perfume_name} (ID: ${finalPerfumeId})`);
      }
    }

    if (!finalPerfumeId) {
      return res.status(400).json({ error: 'No se pudo identificar el perfume' });
    }

    // Verificar si el usuario existe
    const [users] = await db.query('SELECT id FROM usuario WHERE id = ?', [usuario_id]);
    if (users.length === 0) {
      return res.status(400).json({ error: 'Usuario no válido' });
    }

    // Guardar el comentario
    await db.query(
      'INSERT INTO comentario (usuario_id, perfume_id, texto, puntuacion) VALUES (?, ?, ?, ?)',
      [usuario_id, finalPerfumeId, texto, puntuacion || 5]
    );
    
    console.log(`✅ Comentario guardado: Usuario ${usuario_id}, Perfume ${finalPerfumeId}`);
    res.json({ 
      success: true, 
      message: 'Comentario guardado exitosamente',
      perfume_id: finalPerfumeId 
    });
  } catch (err) {
    console.error('❌ Error guardando comentario:', err);
    res.status(500).json({ error: 'Error al guardar comentario en la base de datos' });
  }
});

// ==========================================
//  RUTAS DE WISHLIST
// ==========================================

app.post('/wishlist', async (req, res) => {
  try {
    const { 
      usuario_id, 
      perfume_id, 
      perfume_name, 
      marca, 
      genero, 
      año, 
      main_accords, 
      notas_salida, 
      notas_corazon, 
      notas_base, 
      perfumista 
    } = req.body;
    
    console.log('📥 POST /wishlist - Datos recibidos:', req.body);
    
    // Validar
    if (!usuario_id || !perfume_name) {
      return res.status(400).json({ 
        success: false,
        error: 'usuario_id y perfume_name son requeridos' 
      });
    }
    
    // Insertar usando fecha_agregado
    const [result] = await db.query(
      `INSERT INTO wishlist 
       (usuario_id, perfume_id, perfume_name, marca, genero, año, 
        main_accords, notas_salida, notas_corazon, notas_base, perfumista, fecha_agregado) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())`,
      [
        usuario_id, 
        perfume_id || null, 
        perfume_name, 
        marca || null, 
        genero || null, 
        año || null, 
        JSON.stringify(main_accords) || null,
        notas_salida || null,
        notas_corazon || null,
        notas_base || null,
        perfumista || null
      ]
    );
    
    console.log('✅ Wishlist creada - ID:', result.insertId);
    
    // Devolver el item completo
    const [newItem] = await db.query(
      'SELECT * FROM wishlist WHERE id = ?',
      [result.insertId]
    );
    
    // Asegurar que el frontend reciba created_at
    const itemToReturn = newItem[0] ? {
      ...newItem[0],
      created_at: newItem[0].fecha_agregado || new Date().toISOString()
    } : newItem[0];
    
    res.status(201).json(itemToReturn);
    
  } catch (error) {
    console.error('❌ Error en POST /wishlist:', error);
    res.status(500).json({ 
      success: false,
      error: 'Error interno del servidor'
    });
  }
});

// 2. VERIFICAR SI ESTÁ EN WISHLIST
// En checkInWishlist
app.get('/wishlist/check/:usuarioId/:perfumeIdentifier', async (req, res) => {
  try {
    const { usuarioId, perfumeIdentifier } = req.params;
    
    console.log('🔍 GET /wishlist/check - usuario:', usuarioId, 'perfume:', perfumeIdentifier);
    
    const decodedIdentifier = decodeURIComponent(perfumeIdentifier);
    
    const [result] = await db.query(
      `SELECT id, fecha_agregado FROM wishlist 
       WHERE usuario_id = ? 
       AND (perfume_id = ? OR perfume_name = ? OR perfume_name LIKE ?)`,
      [
        usuarioId, 
        perfumeIdentifier, 
        decodedIdentifier,
        `%${decodedIdentifier}%`
      ]
    );
    
    const exists = result.length > 0;
    
    console.log('✅ Check wishlist - Existe:', exists);
    
    res.json({ 
      exists: exists,
      id: exists ? result[0].id : null,
      fecha_agregado: exists ? result[0].fecha_agregado : null
    });
    
  } catch (error) {
    console.error('❌ Error en GET /wishlist/check:', error);
    res.status(500).json({ 
      exists: false,
      error: 'Error interno del servidor'
    });
  }
});

// 3. ELIMINAR DE WISHLIST
app.delete('/wishlist/:usuarioId/:perfumeIdentifier', async (req, res) => {
  try {
    const { usuarioId, perfumeIdentifier } = req.params;
    
    console.log('🗑️ DELETE /wishlist - usuario:', usuarioId, 'perfume:', perfumeIdentifier);
    
    const decodedIdentifier = decodeURIComponent(perfumeIdentifier);
    
    const [result] = await db.query(
      `DELETE FROM wishlist 
       WHERE usuario_id = ? 
       AND (perfume_id = ? OR perfume_name = ? OR perfume_name LIKE ?)`,
      [
        usuarioId, 
        perfumeIdentifier, 
        decodedIdentifier,
        `%${decodedIdentifier}%`
      ]
    );
    
    if (result.affectedRows === 0) {
      console.log('⚠️ No encontrado para eliminar');
      return res.status(404).json({ 
        success: false,
        error: 'Perfume no encontrado en la wishlist' 
      });
    }
    
    console.log('✅ Wishlist eliminada - Filas afectadas:', result.affectedRows);
    
    res.json({ 
      success: true, 
      message: 'Perfume eliminado de la wishlist' 
    });
    
  } catch (error) {
    console.error('❌ Error en DELETE /wishlist:', error);
    res.status(500).json({ 
      success: false,
      error: 'Error interno del servidor' 
    });
  }
});

// En server.js, modifica la ruta de wishlist
app.get('/wishlist/user/:usuarioId', async (req, res) => {
  try {
    const { usuarioId } = req.params;
    console.log('📋 GET /wishlist/user - usuario:', usuarioId);
    
    // Usa fecha_agregado en lugar de created_at
    const [wishlist] = await db.query(
      'SELECT * FROM wishlist WHERE usuario_id = ? ORDER BY fecha_agregado DESC',
      [usuarioId]
    );
    
    console.log('✅ Wishlist obtenida - Elementos:', wishlist.length);
    
    // Parsear JSON si main_accords está almacenado como string
    const parsedWishlist = wishlist.map(item => {
      try {
        return {
          ...item,
          main_accords: item.main_accords ? JSON.parse(item.main_accords) : [],
          // Asegúrate de que created_at apunte a fecha_agregado para el frontend
          created_at: item.fecha_agregado || item.created_at
        };
      } catch (e) {
        return {
          ...item,
          main_accords: item.main_accords || [],
          created_at: item.fecha_agregado || item.created_at
        };
      }
    });
    
    console.log('📦 Wishlist procesada:', parsedWishlist);
    res.json(parsedWishlist);
    
  } catch (error) {
    console.error('❌ Error en GET /wishlist/user:', error);
    res.status(500).json({ 
      error: 'Error interno del servidor',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// ==========================================
//  RUTAS DE COLECCIÓN (MUY SIMILAR A WISHLIST)
// ==========================================

// 1. AÑADIR A COLECCIÓN
app.post('/collection', async (req, res) => {
  try {
    const { 
      usuario_id, 
      perfume_id, 
      perfume_name, 
      marca, 
      genero, 
      año, 
      main_accords, 
      notas_salida, 
      notas_corazon, 
      notas_base, 
      perfumista,
      fecha_adquisicion 
    } = req.body;
    
    console.log('📥 POST /collection - Datos recibidos:', req.body);
    
    if (!usuario_id) {
      return res.status(400).json({ 
        success: false,
        error: 'usuario_id es requerido' 
      });
    }
    
    if (!perfume_id && !perfume_name) {
      return res.status(400).json({ 
        success: false,
        error: 'perfume_id o perfume_name es requerido' 
      });
    }
    
    // Verificar si ya existe en colección
    const [existing] = await db.query(
      `SELECT id FROM coleccion 
       WHERE usuario_id = ? 
       AND (perfume_id = ? OR perfume_name = ?)`,
      [usuario_id, perfume_id || null, perfume_name]
    );
    
    if (existing.length > 0) {
      return res.status(400).json({ 
        success: false,
        error: 'Este perfume ya está en tu colección' 
      });
    }
    
    // Insertar en la base de datos
    const [result] = await db.query(
      `INSERT INTO coleccion 
       (usuario_id, perfume_id, perfume_name, marca, genero, año, 
        main_accords, notas_salida, notas_corazon, notas_base, perfumista,
        fecha_adquisicion) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        usuario_id, 
        perfume_id || null, 
        perfume_name, 
        marca || null, 
        genero || null, 
        año || null, 
        JSON.stringify(main_accords) || null,
        notas_salida || null,
        notas_corazon || null,
        notas_base || null,
        perfumista || null,
        fecha_adquisicion || new Date().toISOString().split('T')[0]
      ]
    );
    
    console.log('✅ Colección creada - ID:', result.insertId);
    
    res.status(201).json({ 
      success: true, 
      message: 'Perfume añadido a tu colección',
      id: result.insertId 
    });
    
  } catch (error) {
    console.error('❌ Error en POST /collection:', error);
    res.status(500).json({ 
      success: false,
      error: 'Error interno del servidor'
    });
  }
});

// 2. VERIFICAR SI ESTÁ EN COLECCIÓN
app.get('/collection/check/:usuarioId/:perfumeIdentifier', async (req, res) => {
  try {
    const { usuarioId, perfumeIdentifier } = req.params;
    
    console.log('🔍 GET /collection/check - usuario:', usuarioId, 'perfume:', perfumeIdentifier);
    
    const decodedIdentifier = decodeURIComponent(perfumeIdentifier);
    
    const [result] = await db.query(
      `SELECT id FROM coleccion 
       WHERE usuario_id = ? 
       AND (perfume_id = ? OR perfume_name = ? OR perfume_name LIKE ?)`,
      [
        usuarioId, 
        perfumeIdentifier, 
        decodedIdentifier,
        `%${decodedIdentifier}%`
      ]
    );
    
    const exists = result.length > 0;
    
    console.log('✅ Check colección - Existe:', exists);
    
    res.json({ 
      exists: exists,
      id: exists ? result[0].id : null
    });
    
  } catch (error) {
    console.error('❌ Error en GET /collection/check:', error);
    res.status(500).json({ 
      exists: false,
      error: 'Error interno del servidor'
    });
  }
});

// 3. ELIMINAR DE COLECCIÓN
app.delete('/collection/:usuarioId/:perfumeIdentifier', async (req, res) => {
  try {
    const { usuarioId, perfumeIdentifier } = req.params;
    
    console.log('🗑️ DELETE /collection - usuario:', usuarioId, 'perfume:', perfumeIdentifier);
    
    const decodedIdentifier = decodeURIComponent(perfumeIdentifier);
    
    const [result] = await db.query(
      `DELETE FROM coleccion 
       WHERE usuario_id = ? 
       AND (perfume_id = ? OR perfume_name = ? OR perfume_name LIKE ?)`,
      [
        usuarioId, 
        perfumeIdentifier, 
        decodedIdentifier,
        `%${decodedIdentifier}%`
      ]
    );
    
    if (result.affectedRows === 0) {
      console.log('⚠️ No encontrado para eliminar');
      return res.status(404).json({ 
        success: false,
        error: 'Perfume no encontrado en la colección' 
      });
    }
    
    console.log('✅ Colección eliminada - Filas afectadas:', result.affectedRows);
    
    res.json({ 
      success: true, 
      message: 'Perfume eliminado de la colección' 
    });
    
  } catch (error) {
    console.error('❌ Error en DELETE /collection:', error);
    res.status(500).json({ 
      success: false,
      error: 'Error interno del servidor' 
    });
  }
});

// 4. OBTENER COLECCIÓN DEL USUARIO
app.get('/collection/user/:usuarioId', async (req, res) => {
  try {
    const { usuarioId } = req.params;
    
    console.log('📋 GET /collection/user - usuario:', usuarioId);
    
    const [collection] = await db.query(
      'SELECT * FROM coleccion WHERE usuario_id = ? ORDER BY fecha_adquisicion DESC',
      [usuarioId]
    );
    
    // Parsear JSON si main_accords está almacenado como string
    const parsedCollection = collection.map(item => ({
      ...item,
      main_accords: item.main_accords ? JSON.parse(item.main_accords) : []
    }));
    
    console.log('✅ Colección obtenida - Elementos:', parsedCollection.length);
    
    res.json(parsedCollection);
    
  } catch (error) {
    console.error('❌ Error en GET /collection/user:', error);
    res.status(500).json({ 
      error: 'Error interno del servidor' 
    });
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
    
    sql += " LIMIT 50";

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
    const queryId = "SELECT id, marca_id, acordes_principales FROM perfume WHERE nombre LIKE ? OR nombre LIKE ? LIMIT 1";
    const [originals] = await db.query(queryId, [`%${vars[0]}%`, `%${vars[1]}%`]);

    if (originals.length === 0) return res.status(404).json({ similares: [] });
    const original = originals[0];

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
//  CREAR TABLAS SI NO EXISTEN
// ==========================================

// Función para crear las tablas si no existen
const createTablesIfNotExist = async () => {
  try {
    // Tabla wishlist
    await db.query(`
      CREATE TABLE IF NOT EXISTS wishlist (
        id INT AUTO_INCREMENT PRIMARY KEY,
        usuario_id INT NOT NULL,
        perfume_id VARCHAR(255),
        perfume_name VARCHAR(255) NOT NULL,
        marca VARCHAR(255),
        genero VARCHAR(50),
        año VARCHAR(20),
        main_accords TEXT,
        notas_salida TEXT,
        notas_corazon TEXT,
        notas_base TEXT,
        perfumista VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (usuario_id) REFERENCES usuario(id) ON DELETE CASCADE,
        INDEX idx_usuario_perfume (usuario_id, perfume_name)
      )
    `);
    console.log('✅ Tabla wishlist verificada/creada');
    
    // Tabla coleccion
    await db.query(`
      CREATE TABLE IF NOT EXISTS coleccion (
        id INT AUTO_INCREMENT PRIMARY KEY,
        usuario_id INT NOT NULL,
        perfume_id VARCHAR(255),
        perfume_name VARCHAR(255) NOT NULL,
        marca VARCHAR(255),
        genero VARCHAR(50),
        año VARCHAR(20),
        main_accords TEXT,
        notas_salida TEXT,
        notas_corazon TEXT,
        notas_base TEXT,
        perfumista VARCHAR(255),
        fecha_adquisicion DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (usuario_id) REFERENCES usuario(id) ON DELETE CASCADE,
        INDEX idx_usuario_perfume (usuario_id, perfume_name)
      )
    `);
    console.log('✅ Tabla coleccion verificada/creada');
    
  } catch (error) {
    console.error('❌ Error creando tablas:', error);
  }
};

// ==========================================
//  RUTA DE PRUEBA PARA VERIFICAR SERVIDOR
// ==========================================
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    message: 'Servidor Node.js funcionando',
    timestamp: new Date().toISOString(),
    routes: {
      auth: ['POST /register', 'POST /login'],
      comments: ['GET /comentarios/:id', 'POST /comentarios'],
      wishlist: ['POST /wishlist', 'GET /wishlist/check/:userId/:perfumeId', 'DELETE /wishlist/:userId/:perfumeId', 'GET /wishlist/user/:userId'],
      collection: ['POST /collection', 'GET /collection/check/:userId/:perfumeId', 'DELETE /collection/:userId/:perfumeId', 'GET /collection/user/:userId'],
      perfumes: ['GET /perfumes/marcas', 'GET /perfumes/marca/:nombre', 'GET /perfumes/search', 'GET /perfumes/similares']
    }
  });
});

// ==========================================
//  INICIAR SERVIDOR
// ==========================================

// Crear tablas antes de iniciar
createTablesIfNotExist().then(() => {
  const PORT = 3001;
  app.listen(PORT, () => {
    console.log(`🚀 Servidor backend completo corriendo en http://localhost:${PORT}`);
    console.log(`📝 Rutas activas:`);
    console.log(`   POST /wishlist - Añadir a wishlist`);
    console.log(`   GET  /wishlist/check/:userId/:perfumeId - Verificar wishlist`);
    console.log(`   DELETE /wishlist/:userId/:perfumeId - Eliminar de wishlist`);
    console.log(`   GET  /wishlist/user/:userId - Obtener wishlist del usuario`);
    console.log(`   POST /collection - Añadir a colección`);
    console.log(`   GET  /collection/check/:userId/:perfumeId - Verificar colección`);
    console.log(`   DELETE /collection/:userId/:perfumeId - Eliminar de colección`);
    console.log(`   GET  /collection/user/:userId - Obtener colección del usuario`);
    console.log(`   GET  /health - Verificar estado del servidor`);
  });
}).catch(err => {
  console.error('❌ Error al crear tablas:', err);
  process.exit(1);
});