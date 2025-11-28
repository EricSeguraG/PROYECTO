// backend/db.js
const mysql = require('mysql2');

const db = mysql.createPool({
  host: 'localhost',
  user: 'root',
  password: 'root', 
  database: 'essence',          
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

db.getConnection((err, connection) => {
  if (err) console.error('❌ Error conectando a BD:', err.code);
  else {
    console.log('✅ Conectado a la Base de Datos ESSENCE');
    connection.release();
  }
});

module.exports = db.promise();