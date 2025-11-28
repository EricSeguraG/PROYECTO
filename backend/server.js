const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const db = require('./db');

const app = express();
app.use(cors());
app.use(express.json());

// --- REGISTRO ---
app.post('/register', async (req, res) => {
  const { username, password, name, lastname } = req.body;
  
  // Validación básica en el servidor
  if (!username || !password || !name) {
    return res.status(400).json({ error: 'Faltan datos obligatorios' });
  }

  try {
    const [existing] = await db.query('SELECT * FROM usuario WHERE username = ?', [username]);
    if (existing.length > 0) return res.status(400).json({ error: 'El usuario ya existe' });

    const salt = await bcrypt.genSalt(10);
    const hash = await bcrypt.hash(password, salt);

    // Guardamos en BD
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

// --- LOGIN (IMPORTANTE: Mapeo de datos) ---
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

    // AQUÍ ESTÁ EL TRUCO: Enviamos los datos con los nombres que React espera
    res.json({
      id: user.id,
      username: user.username,
      name: user.nombre,       // Traducimos 'nombre' de BD a 'name' de React
      lastname: user.apellido  // Traducimos 'apellido' de BD a 'lastname' de React
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error del servidor' });
  }
});

app.listen(5000, () => console.log('🚀 Servidor listo en puerto 5000'));